// SPDX-License-Identifier: GPL-3.0-only
//! The audio tower, and the projector that hands its output to the decoder.
//!
//! The tower is Whisper's encoder: two convolutions over the spectrogram, the
//! second halving it to `max_source_positions` frames, learnt positions, and
//! pre-norm transformer layers attending over the whole chunk. Chunks are a
//! batch — they never attend to each other.

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, PaddingConfig1d};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::gelu_approximate;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;

use crate::voxtral::config::{Activation, EncoderConfig, VoxtralConfig};
use crate::voxtral::linear_config;
use crate::voxtral::model::Taps;

/// Whisper's layer norms, at PyTorch's default epsilon.
const LAYER_NORM_EPS: f64 = 1e-5;

/// `torch.finfo(torch.float16).max - 1000`, where the encoder clamps in f16.
const F16_CLAMP: f64 = 64_504.0;

#[derive(Module, Debug)]
struct EncoderAttention {
    q_proj: Linear,
    /// Whisper's key projection alone has no bias.
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl EncoderAttention {
    fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let d = cfg.hidden_size;
        Self {
            q_proj: linear_config(d, d).init(device),
            k_proj: linear_config(d, d).with_bias(false).init(device),
            v_proj: linear_config(d, d).init(device),
            out_proj: linear_config(d, d).init(device),
            num_heads: cfg.num_attention_heads,
            head_dim: d / cfg.num_attention_heads,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let [batch, len, _] = xs.dims();
        // (B, L, H * D) -> (B, H, L, D)
        let split = |xs: Tensor<3>| {
            xs.reshape([batch, len, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = split(self.q_proj.forward(xs.clone()));
        let k = split(self.k_proj.forward(xs.clone()));
        let v = split(self.v_proj.forward(xs));
        // Every frame attends to every frame of its chunk; the default scale
        // is the `1/sqrt(head_dim)` Whisper applies to the queries.
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        let out = attention(q, k, v, None, None, options)
            .swap_dims(1, 2)
            .reshape([batch, len, self.num_heads * self.head_dim]);
        self.out_proj.forward(out)
    }
}

#[derive(Module, Debug)]
struct EncoderLayer {
    self_attn: EncoderAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    #[module(skip)]
    act: Activation,
}

impl EncoderLayer {
    fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let norm = || {
            LayerNormConfig::new(cfg.hidden_size)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device)
        };
        Self {
            self_attn: EncoderAttention::init(cfg, device),
            self_attn_layer_norm: norm(),
            fc1: linear_config(cfg.hidden_size, cfg.intermediate_size).init(device),
            fc2: linear_config(cfg.intermediate_size, cfg.hidden_size).init(device),
            final_layer_norm: norm(),
            act: cfg.activation_function,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let xs = xs.clone()
            + self
                .self_attn
                .forward(self.self_attn_layer_norm.forward(xs));
        let hidden = self.final_layer_norm.forward(xs.clone());
        let hidden = self.fc2.forward(self.act.forward(self.fc1.forward(hidden)));
        let xs = xs + hidden;
        // Whisper's guard against f16 overflow, which candle and the Hugging
        // Face implementation both apply: an encoder activation past f16's
        // range would otherwise become an infinity and poison the attention of
        // every layer after it. bf16 and f32 have the range, and skip it.
        if xs.dtype() == DType::F16 {
            xs.clamp(-F16_CLAMP, F16_CLAMP)
        } else {
            xs
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// The learnt positions, `(max_source_positions, hidden_size)`.
    embed_positions: Embedding,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
}

impl AudioEncoder {
    pub fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let d = cfg.hidden_size;
        Self {
            conv1: Conv1dConfig::new(cfg.num_mel_bins, d, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            conv2: Conv1dConfig::new(d, d, 3)
                .with_stride(2)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            embed_positions: EmbeddingConfig::new(cfg.max_source_positions, d).init(device),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| EncoderLayer::init(cfg, device))
                .collect(),
            layer_norm: LayerNormConfig::new(d)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
        }
    }

    /// `features` `(chunks, num_mel_bins, 2 * max_source_positions)` to
    /// `(chunks, max_source_positions, hidden_size)`.
    pub fn forward(&self, features: Tensor<3>, taps: &mut Taps) -> Tensor<3> {
        let xs = self.conv1.forward(features);
        taps.record("enc.conv1", &xs);
        // The tanh approximation after the convolutions, as candle computes
        // it (`Tensor::gelu`); the layers use the exact GELU.
        let xs = self.conv2.forward(gelu_approximate(xs));
        taps.record("enc.conv2", &xs);
        let xs = gelu_approximate(xs).swap_dims(1, 2);
        let [_, frames, d] = xs.dims();
        let positions = self
            .embed_positions
            .weight
            .val()
            .narrow(0, 0, frames)
            .reshape([1, frames, d]);
        let mut xs = xs + positions;
        taps.record("enc.embed", &xs);
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(xs);
            taps.record_with(|| format!("enc.layer.{i}"), &xs);
        }
        let xs = self.layer_norm.forward(xs);
        taps.record("enc.norm", &xs);
        xs
    }
}

/// Folds `frames_per_token` encoder frames into one decoder-sized embedding.
#[derive(Module, Debug)]
pub struct Projector {
    linear_1: Linear,
    linear_2: Linear,
    #[module(skip)]
    act: Activation,
    /// Values per folded token: the encoder width times the frames folded.
    fold: usize,
}

impl Projector {
    pub fn init(cfg: &VoxtralConfig, device: &Device) -> Self {
        let fold = cfg.audio_config.intermediate_size;
        let hidden = cfg.text_config.hidden_size;
        Self {
            linear_1: linear_config(fold, hidden).with_bias(false).init(device),
            linear_2: linear_config(hidden, hidden).with_bias(false).init(device),
            act: cfg.projector_hidden_act,
            fold,
        }
    }

    /// `(chunks, frames, hidden)` encoder output to `(tokens, text_hidden)`,
    /// consecutive frames of a chunk folded together.
    pub fn forward(&self, encoded: Tensor<3>) -> Tensor<2> {
        let n = encoded.shape().num_elements() / self.fold;
        let folded = encoded.reshape([n, self.fold]);
        self.linear_2
            .forward(self.act.forward(self.linear_1.forward(folded)))
    }
}
