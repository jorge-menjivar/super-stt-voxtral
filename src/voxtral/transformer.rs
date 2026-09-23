// SPDX-License-Identifier: GPL-3.0-only
//! The text decoder: a Llama stack — pre-norm, SwiGLU MLP, rotary positions,
//! grouped-query attention.
//!
//! Adapted from the Qwen3 decoder of the Burn fork's `qwen3-tts` example (see
//! the provenance note in [`crate::voxtral`]), which is the same stack plus
//! per-head query/key norms, layer scales and sliding windows; those are gone.
//!
//! The parameters live in [`Transformer`]; everything that changes while
//! generating — the rotary tables and the per-layer key/value caches — lives
//! in [`TransformerState`], one per transcription.
//!
//! The caches are preallocated to the longest sequence the transcription can
//! reach, and every decoding step attends to all of them through a mask
//! hiding the positions after its own. That makes every step the same
//! computation on tensors of the same shapes, whatever its position: CubeCL
//! compiles and tunes its kernels per shape, and a cache grown a token at a
//! time would have it compile a fresh set for every token of the first
//! transcription — over a minute of stalls on an 11-second clip.

use burn::nn::{Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, IndexingUpdateOp};

use crate::voxtral::config::{Activation, TextConfig};
use crate::voxtral::linear_config;
use crate::voxtral::model::Taps;

/// Rotary tables, `(positions, head_dim)`.
///
/// The rotation pairs channel `i` with `i + D / 2` — Llama's `rotate_half`:
/// `out = x * cos + rotate_half(x) * sin`. The tables are stored full width
/// with the sign of the rotated half folded into `sin`, so the rotation is one
/// gather along the channels and arithmetic the fusion folds into the
/// surrounding kernels.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor<2>,
    sin: Tensor<2>,
    /// The channel each output channel is rotated with.
    rotate: Tensor<1, Int>,
}

impl RotaryEmbedding {
    /// Tables for positions `0..positions`.
    ///
    /// The frequencies are computed as candle computes them — `theta` narrowed
    /// to f32 first, then `theta^(2i / D)` in f32 — which is what keeps the
    /// tables bit-identical to the ones the candle backend ran with.
    fn new(cfg: &TextConfig, positions: usize, dtype: DType, device: &Device) -> Self {
        let dim = cfg.head_dim();
        let half = dim / 2;
        #[allow(clippy::cast_possible_truncation)]
        let theta = cfg.rope_theta as f32;
        #[allow(clippy::cast_precision_loss)]
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions * dim);
        let mut sin = Vec::with_capacity(positions * dim);
        for pos in 0..positions {
            #[allow(clippy::cast_precision_loss)]
            let angles: Vec<f32> = inv_freq.iter().map(|freq| pos as f32 * freq).collect();
            // Both halves see the same angle; the first half subtracts its
            // rotated partner.
            cos.extend(angles.iter().map(|a| a.cos()));
            cos.extend(angles.iter().map(|a| a.cos()));
            sin.extend(angles.iter().map(|a| -a.sin()));
            sin.extend(angles.iter().map(|a| a.sin()));
        }
        let rotate: Vec<i64> = (half..dim)
            .chain(0..half)
            .map(|i| i64::try_from(i).expect("a head dimension fits in i64"))
            .collect();
        let shape = [positions, dim];
        Self {
            cos: Tensor::<2>::from_data(TensorData::new(cos, shape), device).cast(dtype),
            sin: Tensor::<2>::from_data(TensorData::new(sin, shape), device).cast(dtype),
            rotate: Tensor::<1, Int>::from_data(TensorData::new(rotate, [dim]), device),
        }
    }

    /// The rows for `seq_len` positions from `offset`, broadcastable over
    /// `(B, H, L, D)`. Sliced once per forward and shared by every layer.
    fn slice(&self, offset: usize, seq_len: usize) -> RotarySlice {
        let dim = self.cos.dims()[1];
        let rows = |table: &Tensor<2>| {
            table
                .clone()
                .narrow(0, offset, seq_len)
                .reshape([1, 1, seq_len, dim])
        };
        RotarySlice {
            cos: rows(&self.cos),
            sin: rows(&self.sin),
            rotate: self.rotate.clone(),
        }
    }

    /// The rows for the one position held by `pos`, looked up on the device
    /// so the pass does not depend on the position's value.
    fn gather(&self, pos: &Tensor<1, Int>) -> RotarySlice {
        let dim = self.cos.dims()[1];
        let row = |table: &Tensor<2>| table.clone().select(0, pos.clone()).reshape([1, 1, 1, dim]);
        RotarySlice {
            cos: row(&self.cos),
            sin: row(&self.sin),
            rotate: self.rotate.clone(),
        }
    }
}

/// The rotary tables narrowed to the positions of one forward pass.
#[derive(Debug, Clone)]
struct RotarySlice {
    cos: Tensor<4>,
    sin: Tensor<4>,
    rotate: Tensor<1, Int>,
}

impl RotarySlice {
    /// Applies RoPE to `xs` `(B, H, L, D)`.
    fn apply(&self, xs: Tensor<4>) -> Tensor<4> {
        let rotated = xs.clone().select(3, self.rotate.clone());
        xs * self.cos.clone() + rotated * self.sin.clone()
    }
}

/// Keys and values of one attention layer, `(1, Hkv, capacity, D)` each,
/// written in place.
#[derive(Debug, Clone)]
struct KvCache {
    k: Tensor<4>,
    v: Tensor<4>,
}

impl KvCache {
    /// Stores `k` and `v` `(B, Hkv, L, D)` at positions `offset..offset + L`
    /// and returns the cache up to them.
    fn write(
        &mut self,
        keys: Tensor<4>,
        values: Tensor<4>,
        offset: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let [batch, heads, len, dim] = keys.dims();
        let at = [0..batch, 0..heads, offset..offset + len, 0..dim];
        // The state holds the only reference to the buffers, so the
        // assignments write into them rather than into copies.
        self.k.inplace(|cache| cache.slice_assign(at.clone(), keys));
        self.v.inplace(|cache| cache.slice_assign(at, values));
        (
            self.k.clone().narrow(2, 0, offset + len),
            self.v.clone().narrow(2, 0, offset + len),
        )
    }

    /// Stores `k` and `v` `(B, Hkv, 1, D)` at the position held by `pos` and
    /// returns the whole cache, for the caller to mask.
    fn write_at(
        &mut self,
        k: Tensor<4>,
        v: Tensor<4>,
        pos: &Tensor<1, Int>,
    ) -> (Tensor<4>, Tensor<4>) {
        self.k
            .inplace(|cache| cache.select_assign(2, pos.clone(), k, IndexingUpdateOp::Assign));
        self.v
            .inplace(|cache| cache.select_assign(2, pos.clone(), v, IndexingUpdateOp::Assign));
        (self.k.clone(), self.v.clone())
    }
}

/// Where a forward pass puts its tokens and how far it attends.
#[derive(Clone, Copy)]
enum Step<'a> {
    /// Tokens at `offset..offset + L`, attending causally to the cache up to
    /// them.
    Prefill { offset: usize },
    /// One token at the position held by `pos`, attending to the whole cache
    /// through the additive `mask` `(1, 1, 1, capacity)`.
    Decode {
        pos: &'a Tensor<1, Int>,
        mask: &'a Tensor<4>,
    },
}

/// What a generation keeps between forward passes.
#[derive(Debug, Clone)]
pub struct TransformerState {
    rotary: RotaryEmbedding,
    caches: Vec<KvCache>,
    /// `0..capacity`, what a token's position is compared with to mask the
    /// positions after it.
    positions: Tensor<1, Int>,
    capacity: usize,
}

impl TransformerState {
    /// A state for a sequence of at most `capacity` tokens.
    pub fn new(cfg: &TextConfig, capacity: usize, dtype: DType, device: &Device) -> Self {
        let shape = [1, cfg.num_key_value_heads, capacity, cfg.head_dim()];
        let capacity_i64 = i64::try_from(capacity).expect("a capacity fits in i64");
        Self {
            rotary: RotaryEmbedding::new(cfg, capacity, dtype, device),
            caches: (0..cfg.num_hidden_layers)
                .map(|_| KvCache {
                    k: Tensor::zeros(shape, (device, dtype)),
                    v: Tensor::zeros(shape, (device, dtype)),
                })
                .collect(),
            positions: Tensor::arange(0..capacity_i64, device),
            capacity,
        }
    }

    /// Additive mask `(1, 1, 1, capacity)` for one query at the position held
    /// by `pos`: `-1e9` on the positions after it, whose keys are zeros or
    /// left over. In f32, where `exp(-1e9)` is exactly zero.
    fn mask(&self, pos: &Tensor<1, Int>) -> Tensor<4> {
        (self.positions.clone() - pos.clone())
            .greater_elem(0)
            .float()
            .cast(DType::F32)
            .mul_scalar(-1e9)
            .reshape([1, 1, 1, self.capacity])
    }
}

#[derive(Module, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    #[module(skip)]
    act: Activation,
}

impl Mlp {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let linear = |d_in, d_out| linear_config(d_in, d_out).with_bias(false).init(device);
        Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size),
            act: cfg.hidden_act,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let gate = self.act.forward(self.gate_proj.forward(xs.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(xs))
    }
}

/// Repeats each key/value head `n_rep` times: the grouped-query expansion.
fn repeat_kv(xs: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return xs;
    }
    let [b, h, l, d] = xs.dims();
    xs.unsqueeze_dim::<5>(2)
        .expand([b, h, n_rep, l, d])
        .reshape([b, h * n_rep, l, d])
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let head_dim = cfg.head_dim();
        let linear = |d_in, d_out| {
            linear_config(d_in, d_out)
                .with_bias(cfg.attention_bias)
                .init(device)
        };
        Self {
            q_proj: linear(cfg.hidden_size, cfg.num_attention_heads * head_dim),
            k_proj: linear(cfg.hidden_size, cfg.num_key_value_heads * head_dim),
            v_proj: linear(cfg.hidden_size, cfg.num_key_value_heads * head_dim),
            o_proj: linear(cfg.num_attention_heads * head_dim, cfg.hidden_size),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        xs: Tensor<3>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let [batch, len, _] = xs.dims();
        // (B, L, H * D) -> (B, H, L, D)
        let split = |xs: Tensor<3>, heads: usize| {
            xs.reshape([batch, len, heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = split(self.q_proj.forward(xs.clone()), self.num_heads);
        let k = split(self.k_proj.forward(xs.clone()), self.num_kv_heads);
        let v = split(self.v_proj.forward(xs), self.num_kv_heads);
        let q = rotary.apply(q);
        let k = rotary.apply(k);

        let out = match step {
            Step::Decode { pos, mask } => {
                let (k, v) = cache.write_at(k, v, pos);
                self.decode(q, k, v, mask)
            }
            Step::Prefill { offset } => {
                let (k, v) = cache.write(k, v, offset);
                let groups = self.num_heads / self.num_kv_heads;
                // One fused kernel rather than a matmul/softmax/matmul chain.
                // The causal flag aligns on the bottom-right corner, which is
                // what a query attending to a whole key/value cache needs.
                let options = AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: true,
                };
                attention(
                    q,
                    repeat_kv(k, groups),
                    repeat_kv(v, groups),
                    None,
                    None,
                    options,
                )
                .swap_dims(1, 2)
                .reshape([batch, len, self.num_heads * self.head_dim])
            }
        };
        self.o_proj.forward(out)
    }

    /// Attention of one query `(B, H, 1, D)` over the whole cache `(B, Hkv,
    /// capacity, D)`, `mask` `(1, 1, 1, capacity)` added to the scores.
    ///
    /// The attention op has no kernel for a single query and falls back to a
    /// dozen small ones on top of the copies expanding the keys and values to
    /// every head. Grouping the query heads by the key/value head they share
    /// turns the expansion into a reshape: `(B, Hkv, G, D) @ (B, Hkv, D, L)`
    /// scores every head at once, in head order.
    fn decode(&self, q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, mask: &Tensor<4>) -> Tensor<3> {
        let [b, _, _, _] = q.dims();
        let dtype = q.dtype();
        let groups = self.num_heads / self.num_kv_heads;
        let q = q.reshape([b, self.num_kv_heads, groups, self.head_dim]);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // The softmax runs in f32, as candle's does.
        let scores = q.matmul(k.transpose()).mul_scalar(scale).cast(DType::F32) + mask.clone();
        let probs = burn::tensor::activation::softmax(scores, 3).cast(dtype);
        probs
            .matmul(v)
            .reshape([b, 1, self.num_heads * self.head_dim])
    }
}

#[derive(Module, Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let norm = || {
            RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device)
        };
        Self {
            self_attn: Attention::init(cfg, device),
            mlp: Mlp::init(cfg, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
        }
    }

    fn forward(
        &self,
        xs: Tensor<3>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let xs = xs.clone()
            + self
                .self_attn
                .forward(self.input_layernorm.forward(xs), rotary, cache, step);
        xs.clone() + self.mlp.forward(self.post_attention_layernorm.forward(xs))
    }
}

/// The decoder layers and the final norm. Works on embeddings rather than
/// token ids, since the prompt's embeddings have the audio spliced in.
#[derive(Module, Debug)]
pub struct Transformer {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Transformer {
    pub fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            layers: (0..cfg.num_hidden_layers)
                .map(|_| DecoderLayer::init(cfg, device))
                .collect(),
            norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// Runs `xs` `(B, L, hidden)`, whose first token sits at `offset`, and
    /// returns the normalized hidden states. Their keys and values are written
    /// into `state` at their positions.
    pub fn forward(
        &self,
        xs: Tensor<3>,
        offset: usize,
        state: &mut TransformerState,
        taps: &mut Taps,
    ) -> Tensor<3> {
        let [_, seq_len, _] = xs.dims();
        assert!(
            offset + seq_len <= state.capacity,
            "positions {offset}..{} are past the {} the state holds",
            offset + seq_len,
            state.capacity
        );
        let rotary = state.rotary.slice(offset, seq_len);
        let step = Step::Prefill { offset };
        let mut xs = xs;
        for (i, (layer, cache)) in self.layers.iter().zip(&mut state.caches).enumerate() {
            xs = layer.forward(xs, &rotary, cache, step);
            taps.record_with(|| format!("dec.layer.{i}"), &xs);
        }
        let xs = self.norm.forward(xs);
        taps.record("dec.norm", &xs);
        xs
    }

    /// Runs one token `xs` `(B, 1, hidden)` at the position held by `pos`,
    /// which is never read on the host: the pass is the same for every
    /// position, so its kernels are compiled and tuned once.
    pub fn forward_at(
        &self,
        xs: Tensor<3>,
        pos: &Tensor<1, Int>,
        state: &mut TransformerState,
    ) -> Tensor<3> {
        let rotary = state.rotary.gather(pos);
        let mask = state.mask(pos);
        let step = Step::Decode { pos, mask: &mask };
        let mut xs = xs;
        for (layer, cache) in self.layers.iter().zip(&mut state.caches) {
            xs = layer.forward(xs, &rotary, cache, step);
        }
        self.norm.forward(xs)
    }
}
