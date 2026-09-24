// SPDX-License-Identifier: GPL-3.0-only
//! The `config.json` of a Voxtral checkpoint.
//!
//! Every field has a default, the Voxtral Mini 3B value, so a config that
//! leaves a key out loads the way the Hugging Face classes would load it. Only
//! the two sections are required: a file without them is not a Voxtral config.

use burn::prelude::*;
use burn::tensor::activation::{gelu, relu, silu};
use serde::Deserialize;

/// An activation named in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// The exact, `erf`-based GELU, which is what `transformers` means by
    /// `"gelu"`.
    Gelu,
    Relu,
    Silu,
}

impl Activation {
    pub fn forward<const D: usize>(self, xs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Gelu => gelu(xs),
            Self::Relu => relu(xs),
            Self::Silu => silu(xs),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub audio_config: EncoderConfig,
    pub text_config: TextConfig,
    /// The placeholder the projected audio replaces in the prompt.
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    #[serde(default = "default_projector_act")]
    pub projector_hidden_act: Activation,
}

fn default_audio_token_id() -> u32 {
    24
}

fn default_projector_act() -> Activation {
    Activation::Gelu
}

/// The audio tower.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    /// Encoder frames per chunk: the length of the learnt position table.
    pub max_source_positions: usize,
    pub activation_function: Activation,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            num_mel_bins: 128,
            max_source_positions: 1500,
            activation_function: Activation::Gelu,
        }
    }
}

/// The text decoder, a Llama configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Absent means `hidden_size / num_attention_heads`, see
    /// [`Self::head_dim`].
    #[serde(rename = "head_dim")]
    pub explicit_head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: Activation,
    /// Whether the query, key, value and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether the output head reuses the token embeddings.
    pub tie_word_embeddings: bool,
    /// Only plain rotary positions are implemented; a checkpoint asking for a
    /// scaled variant is refused rather than run with the wrong positions.
    pub rope_scaling: Option<serde_json::Value>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131_072,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 30,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            explicit_head_dim: None,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000_000.0,
            max_position_embeddings: 131_072,
            hidden_act: Activation::Silu,
            attention_bias: false,
            tie_word_embeddings: false,
            rope_scaling: None,
        }
    }
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl VoxtralConfig {
    /// Parse a `config.json`, and check the shapes the port depends on agree
    /// with each other.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> anyhow::Result<()> {
        let a = &self.audio_config;
        let t = &self.text_config;
        anyhow::ensure!(
            a.num_attention_heads > 0 && a.hidden_size.is_multiple_of(a.num_attention_heads),
            "audio hidden_size {} is not a whole number of {} heads",
            a.hidden_size,
            a.num_attention_heads
        );
        // The projector reads `intermediate_size` values at a time out of the
        // encoder output, which is how four frames become one embedding.
        anyhow::ensure!(
            a.hidden_size > 0 && a.intermediate_size.is_multiple_of(a.hidden_size),
            "audio intermediate_size {} is not a whole number of {}-wide frames",
            a.intermediate_size,
            a.hidden_size
        );
        anyhow::ensure!(
            t.num_key_value_heads > 0
                && t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            "{} query heads cannot be grouped over {} key/value heads",
            t.num_attention_heads,
            t.num_key_value_heads
        );
        anyhow::ensure!(
            t.rope_scaling
                .as_ref()
                .is_none_or(serde_json::Value::is_null),
            "rope_scaling {} is not implemented",
            t.rope_scaling
                .as_ref()
                .map_or_else(String::new, ToString::to_string)
        );
        anyhow::ensure!(
            t.head_dim().is_multiple_of(2),
            "rotary positions need an even head_dim, not {}",
            t.head_dim()
        );
        Ok(())
    }

    /// Encoder frames folded into one audio embedding: four for every
    /// published checkpoint.
    pub fn frames_per_audio_token(&self) -> usize {
        self.audio_config.intermediate_size / self.audio_config.hidden_size
    }

    /// Encoder frames per 30-second chunk of audio: the two convolutions halve
    /// the 3000 mel frames to `max_source_positions`.
    pub fn encoder_frames_per_chunk(&self) -> usize {
        self.audio_config.max_source_positions
    }

    /// Placeholder tokens one chunk of audio fills in the prompt.
    pub fn audio_tokens_per_chunk(&self) -> usize {
        self.encoder_frames_per_chunk() / self.frames_per_audio_token()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_voxtral_mini() {
        let cfg = VoxtralConfig::from_json(r#"{"audio_config":{},"text_config":{}}"#).unwrap();
        assert_eq!(cfg.audio_token_id, 24);
        assert_eq!(cfg.projector_hidden_act, Activation::Gelu);
        assert_eq!(cfg.audio_config.num_mel_bins, 128);
        assert_eq!(cfg.text_config.hidden_act, Activation::Silu);
        assert_eq!(cfg.text_config.head_dim(), 3072 / 32);
        assert!(!cfg.text_config.tie_word_embeddings);
        assert!(!cfg.text_config.attention_bias);
        assert_eq!(cfg.audio_tokens_per_chunk(), 375);
    }

    #[test]
    fn parses_values() {
        let cfg = VoxtralConfig::from_json(
            r#"{
                "audio_token_id": 42,
                "projector_hidden_act": "silu",
                "audio_config": {"hidden_size": 640, "intermediate_size": 2560,
                                 "num_attention_heads": 10, "activation_function": "relu"},
                "text_config": {"vocab_size": 123, "head_dim": 128, "rope_theta": 1e7,
                                "attention_bias": true, "tie_word_embeddings": true}
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.audio_token_id, 42);
        assert_eq!(cfg.projector_hidden_act, Activation::Silu);
        assert_eq!(cfg.audio_config.hidden_size, 640);
        assert_eq!(cfg.audio_config.activation_function, Activation::Relu);
        assert_eq!(cfg.text_config.vocab_size, 123);
        assert_eq!(cfg.text_config.head_dim(), 128);
        assert!((cfg.text_config.rope_theta - 1e7).abs() < 1.0);
        assert!(cfg.text_config.attention_bias);
        assert!(cfg.text_config.tie_word_embeddings);
    }

    /// The candle port read `tie_word_embeddings` out of the `attention_bias`
    /// key. They are two different things — a Llama `attention_bias` puts a
    /// bias on the projections — so each is now read from its own key.
    #[test]
    fn attention_bias_does_not_tie_embeddings() {
        let cfg = VoxtralConfig::from_json(
            r#"{"audio_config":{},"text_config":{"attention_bias": true}}"#,
        )
        .unwrap();
        assert!(cfg.text_config.attention_bias);
        assert!(!cfg.text_config.tie_word_embeddings);
    }

    #[test]
    fn both_sections_are_required() {
        let err = VoxtralConfig::from_json(r#"{"text_config":{}}"#).unwrap_err();
        assert!(err.to_string().contains("audio_config"), "{err}");
        let err = VoxtralConfig::from_json(r#"{"audio_config":{}}"#).unwrap_err();
        assert!(err.to_string().contains("text_config"), "{err}");
    }

    #[test]
    fn an_unknown_activation_is_refused() {
        let err = VoxtralConfig::from_json(
            r#"{"projector_hidden_act":"swish","audio_config":{},"text_config":{}}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("swish"), "{err}");
    }

    #[test]
    fn inconsistent_shapes_are_refused() {
        let err = VoxtralConfig::from_json(
            r#"{"audio_config":{"intermediate_size": 5000},"text_config":{}}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("intermediate_size"), "{err}");
        let err = VoxtralConfig::from_json(
            r#"{"audio_config":{},"text_config":{"num_key_value_heads": 5}}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("grouped"), "{err}");
        let err = VoxtralConfig::from_json(
            r#"{"audio_config":{},"text_config":{"rope_scaling": {"type": "yarn"}}}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("rope_scaling"), "{err}");
    }

    /// The config the manifest downloads for Voxtral Mini, verbatim but for the
    /// keys this port does not read.
    #[test]
    fn parses_the_shipped_mini_config() {
        let cfg = VoxtralConfig::from_json(
            r#"{
              "audio_config": {"activation_function": "gelu", "head_dim": 64, "hidden_size": 1280,
                "intermediate_size": 5120, "max_source_positions": 1500, "num_attention_heads": 20,
                "num_hidden_layers": 32, "num_key_value_heads": 20, "num_mel_bins": 128},
              "audio_token_id": 24, "hidden_size": 3072, "projector_hidden_act": "gelu",
              "text_config": {"attention_bias": false, "head_dim": 128, "hidden_act": "silu",
                "hidden_size": 3072, "intermediate_size": 8192, "max_position_embeddings": 131072,
                "mlp_bias": false, "num_attention_heads": 32, "num_hidden_layers": 30,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-05, "rope_scaling": null,
                "rope_theta": 100000000.0, "sliding_window": null, "vocab_size": 131072},
              "torch_dtype": "bfloat16"
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.text_config.head_dim(), 128);
        assert_eq!(cfg.frames_per_audio_token(), 4);
        assert_eq!(cfg.audio_tokens_per_chunk(), 375);
    }
}
