// SPDX-License-Identifier: GPL-3.0-only
//! candle's Voxtral forward with tap points.
//!
//! Copied from `candle-transformers/src/models/voxtral/{model,voxtral_llama}.rs`
//! at `jorge-menjivar/candle@eef47d5` — the arithmetic verbatim, in the same
//! order — with every layer's output recorded into a [`Taps`]. candle keeps
//! the layers private, so there is no way to observe them without a copy.
//! What makes the copy a reference rather than a second implementation is
//! `main`'s cross-check: its logits and tokens must equal those of candle's own,
//! unmodified `VoxtralForConditionalGeneration`, or the dump is refused.
//!
//! Left out: the training paths (dropout and layer-drop are identities at
//! inference), sampling (the backend decodes greedily), and flash attention
//! (the backend never enabled it).

use std::collections::HashMap;

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Embedding, LayerNorm, Linear, VarBuilder, layer_norm, linear};
use candle_transformers::models::voxtral::model::{
    find_audio_token_positions, replace_audio_tokens,
};
use candle_transformers::models::voxtral::{
    VoxtralConfig, VoxtralEncoderConfig, VoxtralLlamaConfig,
};
use candle_transformers::models::with_tracing;

/// Every tapped tensor, by name, in f32 on the host.
#[derive(Default)]
pub struct Taps {
    pub map: HashMap<String, Tensor>,
}

impl Taps {
    pub fn put(&mut self, name: impl Into<String>, t: &Tensor) -> Result<()> {
        let t = t
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;
        self.map.insert(name.into(), t);
        Ok(())
    }
}

fn safe_clamp(x: &Tensor) -> Result<Tensor> {
    match x.dtype() {
        DType::F16 => {
            let max_val = 64504.0;
            x.clamp(-max_val, max_val)
        }
        _ => Ok(x.clone()),
    }
}

fn activation(name: &str) -> Result<candle_nn::Activation> {
    match name {
        "gelu" => Ok(candle_nn::Activation::Gelu),
        "relu" => Ok(candle_nn::Activation::Relu),
        other => candle_core::bail!("unsupported activation {other}"),
    }
}

struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl EncoderAttention {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        #[allow(clippy::cast_precision_loss)]
        let scaling = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj: linear(embed_dim, embed_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("k_proj"))?,
            v_proj: linear(embed_dim, embed_dim, vb.pp("v_proj"))?,
            out_proj: linear(embed_dim, embed_dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scaling,
        })
    }

    fn reshape_for_scores(&self, x: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        x.reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let q = (self.q_proj.forward(x)? * self.scaling)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = self.reshape_for_scores(&q, seq_len, bsz)?;
        let k = self.reshape_for_scores(&k, seq_len, bsz)?;
        let v = self.reshape_for_scores(&v, seq_len, bsz)?;
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            bsz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj.forward(&attn_output)
    }
}

struct EncoderLayer {
    self_attn: EncoderAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: candle_nn::Activation,
}

impl EncoderLayer {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        Ok(Self {
            self_attn: EncoderAttention::new(cfg, vb.pp("self_attn"))?,
            self_attn_layer_norm: layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?,
            fc1: linear(embed_dim, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, embed_dim, vb.pp("fc2"))?,
            final_layer_norm: layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?,
            activation: activation(&cfg.activation_function)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;
        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.apply(&self.activation)?;
        let x = self.fc2.forward(&x)?;
        let x = (x + residual)?;
        safe_clamp(&x)
    }
}

struct Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Tensor,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
}

impl Encoder {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let conv1 = candle_nn::conv1d(
            cfg.num_mel_bins,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            embed_dim,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let embed_positions = vb.get(
            (cfg.max_source_positions, embed_dim),
            "embed_positions.weight",
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| EncoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layers,
            layer_norm,
        })
    }

    fn forward(&self, input_features: &Tensor, taps: &mut Taps) -> Result<Tensor> {
        let expected_dtype = self.conv1.weight().dtype();
        let input_features = if input_features.dtype() == expected_dtype {
            input_features.clone()
        } else {
            input_features.to_dtype(expected_dtype)?
        };
        let x = self.conv1.forward(&input_features)?;
        taps.put("enc.conv1", &x)?;
        // `Tensor::gelu` is candle's tanh approximation.
        let x = x.gelu()?;
        let x = self.conv2.forward(&x)?;
        taps.put("enc.conv2", &x)?;
        let x = x.gelu()?;
        let x = x.transpose(1, 2)?;
        let seq_len = x.dim(1)?;
        let positions = self.embed_positions.i(..seq_len)?;
        let x = if x.dtype() == positions.dtype() {
            x.broadcast_add(&positions)?
        } else {
            let x_f32 = x.to_dtype(DType::F32)?;
            let result_f32 = x_f32.broadcast_add(&positions)?;
            result_f32.to_dtype(x.dtype())?
        };
        taps.put("enc.embed", &x)?;
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            taps.put(format!("enc.layer.{i}"), &x)?;
        }
        let x = self.layer_norm.forward(&x)?;
        taps.put("enc.norm", &x)?;
        Ok(x)
    }
}

struct Projector {
    linear_1: Linear,
    linear_2: Linear,
    activation: candle_nn::Activation,
}

impl Projector {
    fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: candle_nn::linear_no_bias(
                cfg.audio_config.intermediate_size,
                cfg.text_config.hidden_size,
                vb.pp("linear_1"),
            )?,
            linear_2: candle_nn::linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.hidden_size,
                vb.pp("linear_2"),
            )?,
            activation: activation(&cfg.projector_hidden_act)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = x.apply(&self.activation)?;
        self.linear_2.forward(&x)
    }
}

pub struct LlamaCache {
    masks: HashMap<(usize, usize), Tensor>,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl LlamaCache {
    pub fn new(dtype: DType, config: &VoxtralLlamaConfig, device: &Device) -> Result<Self> {
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        #[allow(clippy::cast_precision_loss)]
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta, device)?;
        #[allow(clippy::cast_possible_truncation)]
        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            masks: HashMap::new(),
            kvs: vec![None; config.num_hidden_layers],
            cos: idx_theta.cos()?.to_dtype(dtype)?,
            sin: idx_theta.sin()?.to_dtype(dtype)?,
            device: device.clone(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask =
                candle_transformers::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }
}

struct CausalSelfAttention {
    q_proj: with_tracing::Linear,
    k_proj: with_tracing::Linear,
    v_proj: with_tracing::Linear,
    o_proj: with_tracing::Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &VoxtralLlamaConfig) -> Result<Self> {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let size_q = head_dim * cfg.num_attention_heads;
        let size_kv = head_dim * cfg.num_key_value_heads;
        Ok(Self {
            q_proj: with_tracing::linear_no_bias(cfg.hidden_size, size_q, vb.pp("q_proj"))?,
            k_proj: with_tracing::linear_no_bias(cfg.hidden_size, size_kv, vb.pp("k_proj"))?,
            v_proj: with_tracing::linear_no_bias(cfg.hidden_size, size_kv, vb.pp("v_proj"))?,
            o_proj: with_tracing::linear_no_bias(size_q, cfg.hidden_size, vb.pp("o_proj"))?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &LlamaCache) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.to_dtype(x.dtype())?;
        let sin = sin.to_dtype(x.dtype())?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;
        // candle trims the cache here when `k.dims()[1]` exceeds
        // `max_position_embeddings` — but that dimension is the head count, so
        // the trim can never fire, and it is left out.
        if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
        }
        cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        let k = candle_transformers::utils::repeat_kv(k, n_rep)?;
        let v = candle_transformers::utils::repeat_kv(v, n_rep)?;
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        #[allow(clippy::cast_precision_loss)]
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape(&[
            b_sz,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ])?;
        self.o_proj.forward(&y)
    }
}

struct Mlp {
    c_fc1: with_tracing::Linear,
    c_fc2: with_tracing::Linear,
    c_proj: with_tracing::Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &VoxtralLlamaConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            c_fc1: with_tracing::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            c_fc2: with_tracing::linear_no_bias(h, i, vb.pp("up_proj"))?,
            c_proj: with_tracing::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }
}

struct Block {
    rms_1: with_tracing::RmsNorm,
    attn: CausalSelfAttention,
    rms_2: with_tracing::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &VoxtralLlamaConfig) -> Result<Self> {
        Ok(Self {
            rms_1: with_tracing::RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attn: CausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            rms_2: with_tracing::RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: with_tracing::RmsNorm,
    lm_head: with_tracing::Linear,
}

impl Llama {
    fn load(vb: VarBuilder, cfg: &VoxtralLlamaConfig) -> Result<Self> {
        let wte =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            with_tracing::Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            with_tracing::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f =
            with_tracing::RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    /// `prefix` names the taps of this pass: layer outputs are only worth
    /// recording for the prefill, whose every position is compared.
    fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut LlamaCache,
        taps: Option<&mut Taps>,
    ) -> Result<Tensor> {
        let mut taps = taps;
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
            if let Some(taps) = &mut taps {
                taps.put(format!("dec.layer.{block_idx}"), &x)?;
            }
        }
        let x = self.ln_f.forward(&x)?;
        if let Some(taps) = &mut taps {
            taps.put("dec.norm", &x)?;
        }
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }
}

/// The whole model, as `VoxtralForConditionalGeneration` assembles it.
pub struct Tapped {
    audio_tower: Encoder,
    projector: Projector,
    language_model: Llama,
    audio_token_id: usize,
    intermediate_size: usize,
}

impl Tapped {
    pub fn new(cfg: &VoxtralConfig, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            audio_tower: Encoder::new(&cfg.audio_config, vb.pp("audio_tower"))?,
            projector: Projector::new(cfg, vb.pp("multi_modal_projector"))?,
            language_model: Llama::load(vb.pp("language_model"), &cfg.text_config)?,
            audio_token_id: cfg.audio_token_id,
            intermediate_size: cfg.audio_config.intermediate_size,
        })
    }

    /// The prefill: audio through the tower and the projector, spliced into
    /// the prompt's embeddings, and the prompt through the decoder. Returns the
    /// logits of the last position, `(1, vocab)`.
    pub fn prefill(
        &self,
        input_ids: &Tensor,
        features: &Tensor,
        cache: &mut LlamaCache,
        taps: &mut Taps,
    ) -> Result<Tensor> {
        let audio = self.audio_tower.forward(features, taps)?;
        let (b, s, h) = audio.dims3()?;
        let audio = audio.reshape((b * s * h / self.intermediate_size, self.intermediate_size))?;
        let audio_embeds = self.projector.forward(&audio)?;
        taps.put("proj", &audio_embeds)?;
        let embeds = self.language_model.wte.forward(input_ids)?;
        let positions = find_audio_token_positions(input_ids, self.audio_token_id)?;
        let embeds = replace_audio_tokens(&embeds, &audio_embeds, &positions, input_ids.device())?;
        taps.put("dec.embeds", &embeds)?;
        self.language_model
            .forward_input_embed(&embeds, 0, cache, Some(taps))
    }

    /// One decoding step: `token` at `index_pos`. Returns `(1, vocab)` logits.
    pub fn step(&self, token: u32, index_pos: usize, cache: &mut LlamaCache) -> Result<Tensor> {
        let device = self.language_model.wte.embeddings().device();
        let input = Tensor::new(&[token], device)?.unsqueeze(0)?;
        let embeds = self.language_model.wte.forward(&input)?;
        self.language_model
            .forward_input_embed(&embeds, index_pos, cache, None)
    }
}
