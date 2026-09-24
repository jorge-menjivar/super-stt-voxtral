// SPDX-License-Identifier: GPL-3.0-only
//! Dump every layer of candle's Voxtral over one clip, for the Burn port to be
//! compared against.
//!
//! ```text
//! voxtral-candle-reference <model-dir> <clip.wav> <out.safetensors> [--dtype f32|f16|bf16] [--lang en]
//! ```
//!
//! The inputs are built exactly as the candle backend built them — the clip
//! padded to whole 30-second chunks, candle's own `extract_features`, the same
//! prompt tokens, the same `config.json` parsing — so the dump is what that
//! backend computed, not an approximation of it. Besides the taps (see
//! [`tapped`]) the file carries what the comparison needs to replay the run:
//! the padded samples (`audio`), the prompt (`input_ids`) and the greedy tokens
//! (`tokens`), with the logits of every step that produced one (`logits.{n}`).
//!
//! Before writing anything, the tapped copy's prefill logits and its greedy
//! tokens are checked against candle's unmodified model. The copy exists only
//! because candle's layers are private; the check is what lets it stand in
//! for the original.

mod tapped;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::voxtral::{
    VoxtralCache, VoxtralConfig, VoxtralEncoderConfig, VoxtralForConditionalGeneration,
    VoxtralGenerationConfig, VoxtralLlamaConfig, audio,
};
use tekken::Tekkenizer;

const SAMPLE_RATE: u32 = 16_000;
const CHUNK_SAMPLES: usize = 480_000;
const MAX_NEW_TOKENS: usize = 1000;
/// candle's end-of-sequence set, from `VoxtralForConditionalGeneration::generate`.
const EOS_TOKENS: [u32; 4] = [2, 128_001, 128_009, 128_256];
const MEL_FILTERS: &[u8] = include_bytes!("../../src/data/melfilters128.bytes");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut dtype = DType::F32;
    let mut lang = "en".to_string();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dtype" => {
                dtype = match it.next().map(String::as_str) {
                    Some("f32") => DType::F32,
                    Some("f16") => DType::F16,
                    Some("bf16") => DType::BF16,
                    other => bail!("--dtype takes f32, f16 or bf16, not {other:?}"),
                }
            }
            "--lang" => lang.clone_from(it.next().context("--lang takes a code")?),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [model_dir, clip, out] = positional.as_slice() else {
        bail!(
            "usage: voxtral-candle-reference <model-dir> <clip.wav> <out.safetensors> [--dtype f32|f16|bf16] [--lang en]"
        );
    };

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    eprintln!("candle on {device:?} in {dtype:?}");

    let samples = pad_to_chunk(&read_wav(clip)?, CHUNK_SAMPLES);
    let filters: Vec<f32> = MEL_FILTERS
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let features = audio::extract_features(&samples, &filters, &device)?;

    let config = load_model_config(&model_dir.join("config.json"))?;
    let tokenizer =
        Tekkenizer::from_file(model_dir.join("tekken.json")).map_err(anyhow::Error::msg)?;
    let input_tokens = prompt(&tokenizer, features.dim(0)?, config.audio_token_id, &lang)?;
    let input_ids = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;

    let mut weights: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    weights.sort();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };

    // candle's own model: the ground truth the copy is held to.
    let original = VoxtralForConditionalGeneration::new(&config, vb.clone())?;
    let started = std::time::Instant::now();
    let mut cache = VoxtralCache::new(true, dtype, &config.text_config, &device)?;
    let original_logits = original.forward(&input_ids, Some(&features), &mut cache, 0)?;
    let original_tokens = original
        .generate(
            &input_ids,
            Some(&features),
            VoxtralGenerationConfig {
                max_new_tokens: MAX_NEW_TOKENS,
                temperature: 0.0,
                top_p: None,
                device: device.clone(),
                cache: Some(VoxtralCache::new(
                    true,
                    dtype,
                    &config.text_config,
                    &device,
                )?),
            },
        )
        .map_err(|e| anyhow::anyhow!("candle generate: {e}"))?;
    let original_tokens = original_tokens[input_tokens.len()..].to_vec();
    eprintln!(
        "original model: {} tokens in {:.1?}",
        original_tokens.len(),
        started.elapsed()
    );
    drop(original);

    let model = tapped::Tapped::new(&config, &vb)?;
    let mut taps = tapped::Taps::default();
    taps.put("mel", &features)?;
    let started = std::time::Instant::now();
    let mut cache = tapped::LlamaCache::new(dtype, &config.text_config, &device)?;
    let mut logits = model.prefill(&input_ids, &features, &mut cache, &mut taps)?;

    let diff = (&logits - &original_logits)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    ensure!(
        diff == 0.0,
        "the tapped copy's prefill logits differ from candle's by up to {diff}; it is not a faithful copy"
    );

    let mut tokens = Vec::new();
    for step in 0..MAX_NEW_TOKENS {
        taps.put(format!("logits.{step}"), &logits.squeeze(0)?)?;
        let token = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;
        tokens.push(token);
        if EOS_TOKENS.contains(&token) || tokens.ends_with(&[0; 5]) {
            break;
        }
        logits = model.step(token, input_tokens.len() + step, &mut cache)?;
    }
    eprintln!(
        "tapped copy: {} tokens in {:.1?}",
        tokens.len(),
        started.elapsed()
    );
    ensure!(
        tokens == original_tokens,
        "the tapped copy generated {tokens:?}, candle generated {original_tokens:?}"
    );

    let text = tokenizer
        .decode(&tokens, tekken::SpecialTokenPolicy::Ignore)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    eprintln!("transcript: {text:?}");

    let cpu = Device::Cpu;
    taps.map
        .insert("audio".into(), Tensor::new(samples.as_slice(), &cpu)?);
    taps.map.insert(
        "input_ids".into(),
        Tensor::new(input_tokens.as_slice(), &cpu)?,
    );
    taps.map
        .insert("tokens".into(), Tensor::new(tokens.as_slice(), &cpu)?);
    candle_core::safetensors::save(&taps.map, out)?;
    eprintln!("wrote {} tensors to {}", taps.map.len(), out.display());
    Ok(())
}

/// Mono samples in `[-1, 1]`, read the way the backend's end-to-end test does.
fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    ensure!(spec.channels == 1, "{} is not mono", path.display());
    ensure!(
        spec.sample_rate == SAMPLE_RATE,
        "{} is {} Hz; the backend is sent {SAMPLE_RATE} Hz",
        path.display(),
        spec.sample_rate
    );
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| f32::from(s) / f32::from(i16::MAX)))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    })
}

/// The candle backend's `pad_to_chunk`, verbatim.
fn pad_to_chunk(audio: &[f32], chunk: usize) -> Vec<f32> {
    if audio.len().is_multiple_of(chunk) {
        audio.to_vec()
    } else {
        let target = ((audio.len() / chunk) + 1) * chunk;
        let mut p = audio.to_vec();
        p.resize(target, 0.0);
        p
    }
}

/// `<s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:<code>[TRANSCRIBE]`, as the
/// candle backend built it.
fn prompt(
    tokenizer: &Tekkenizer,
    chunks: usize,
    audio_token_id: usize,
    lang: &str,
) -> Result<Vec<u32>> {
    let mut tokens = vec![1u32, 3, 25];
    tokens.extend(std::iter::repeat_n(
        u32::try_from(audio_token_id)?,
        chunks * 375,
    ));
    tokens.push(4);
    tokens.extend(
        tokenizer
            .encode(&format!("lang:{lang}"), false, false)
            .map_err(|e| anyhow::anyhow!("encode lang prompt: {e}"))?,
    );
    tokens.push(34);
    Ok(tokens)
}

// The candle backend's `config.json` parsing, verbatim but for the flash
// attention flag, which it never set.

fn load_model_config(config_file: &Path) -> Result<VoxtralConfig> {
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
    let audio_token_id = json
        .get("audio_token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(24);
    Ok(VoxtralConfig {
        audio_config: parse_audio_config(&json)?,
        text_config: parse_text_config(&json)?,
        audio_token_id,
        projector_hidden_act: json
            .get("projector_hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
    })
}

fn parse_audio_config(json: &serde_json::Value) -> Result<VoxtralEncoderConfig> {
    let a = json.get("audio_config").context("Missing audio_config")?;
    let u = |k: &str, d: usize| {
        a.get(k)
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(d)
    };
    let f = |k: &str, d: f64| a.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    Ok(VoxtralEncoderConfig {
        vocab_size: u("vocab_size", 51866),
        hidden_size: u("hidden_size", 1280),
        num_hidden_layers: u("num_hidden_layers", 32),
        num_attention_heads: u("num_attention_heads", 20),
        num_key_value_heads: u("num_key_value_heads", 20),
        intermediate_size: u("intermediate_size", 5120),
        dropout: f("dropout", 0.0),
        attention_dropout: f("attention_dropout", 0.0),
        activation_dropout: f("activation_dropout", 0.0),
        activation_function: a
            .get("activation_function")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
        max_source_positions: u("max_source_positions", 1500),
        layerdrop: f("layerdrop", 0.0),
        initializer_range: f("initializer_range", 0.02),
        scale_embedding: a
            .get("scale_embedding")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        num_mel_bins: u("num_mel_bins", 128),
        head_dim: u("head_dim", 64),
    })
}

fn parse_text_config(json: &serde_json::Value) -> Result<VoxtralLlamaConfig> {
    let t = json.get("text_config").context("Missing text_config")?;
    let u = |k: &str, d: usize| {
        t.get(k)
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(d)
    };
    #[allow(clippy::cast_possible_truncation)]
    let rope_theta = t
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(100_000_000.0) as f32;
    Ok(VoxtralLlamaConfig {
        vocab_size: u("vocab_size", 131_072),
        hidden_size: u("hidden_size", 3072),
        intermediate_size: u("intermediate_size", 8192),
        num_hidden_layers: u("num_hidden_layers", 30),
        num_attention_heads: u("num_attention_heads", 32),
        num_key_value_heads: u("num_key_value_heads", 8),
        head_dim: t
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok()),
        rms_norm_eps: t
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-5),
        rope_theta,
        max_position_embeddings: u("max_position_embeddings", 131_072),
        use_flash_attn: false,
        tie_word_embeddings: t
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}
