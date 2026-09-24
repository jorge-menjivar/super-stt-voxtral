// SPDX-License-Identifier: GPL-3.0-only
//! The engine the `/v1` handlers drive: which device to run on, where the
//! files are, how a request becomes a prompt, and how the tokens that come
//! back become a transcript. The model itself is in [`crate::voxtral`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Error, Result};
use burn::prelude::Device;
use burn::tensor::DType;
use log::{info, warn};
use tekken::Tekkenizer;

use crate::voxtral::audio::{self, CHUNK_SAMPLES, SAMPLE_RATE};
use crate::voxtral::config::VoxtralConfig;
use crate::voxtral::model::{Taps, Voxtral};

/// Longest transcript, in tokens: the cap candle's backend decoded to. It
/// also sizes the key/value caches, see [`Voxtral::generate`].
pub const MAX_NEW_TOKENS: usize = 1000;

/// What the warm-up feeds the decoder, see [`VoxtralEngine::warm_up`]. Any
/// tokens do: only the shapes they run at matter. Three, so the steady state
/// after the first step's compile is reached too.
const WARM_UP_TOKENS: [u32; 3] = [0; 3];

/// `<s>`, `[INST]` and `[BEGIN_AUDIO]`, which open the prompt.
const PROMPT_OPEN: [u32; 3] = [1, 3, 25];
/// `[/INST]`, which closes the audio.
const INST_CLOSE: u32 = 4;
/// `[TRANSCRIBE]`, which asks for a transcript rather than an answer.
const TRANSCRIBE: u32 = 34;

/// The accelerator this build was compiled for, as the manifest names it.
///
/// One build serves one accelerator: Burn's backends are cargo features, and
/// the asset that carries this binary declares the matching `accel`.
const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "wgpu") {
    "wgpu"
} else {
    "cpu"
};

/// Whether [`BUILT_FOR`] is a GPU, which decides the dtype.
const ON_GPU: bool = cfg!(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
));

/// The device this build runs on, and the name `GET /v1/status` reports.
///
/// The daemon sends the accelerator it resolved for the installed asset, or
/// nothing. A build has exactly one backend compiled in, so the request is a
/// cross-check rather than a choice, and a mismatch is worth a line in the log
/// because it means the wrong asset was installed for the host.
pub fn select_device(requested: Option<&str>) -> (Device, &'static str) {
    if let Some(d) = requested.map(str::trim).filter(|s| !s.is_empty())
        && !d.eq_ignore_ascii_case(BUILT_FOR)
    {
        warn!("the daemon asked for {d:?} but this build only has {BUILT_FOR}; using {BUILT_FOR}");
    }
    // One arm per backend, in the order a build that somehow enabled several
    // would prefer them: cargo features are additive, so the arms have to
    // exclude each other by hand.
    #[cfg(feature = "cuda")]
    return (Device::cuda(0), "cuda");
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    return (Device::rocm(0), "rocm");
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    return (
        Device::vulkan(burn::prelude::DeviceKind::DefaultDevice),
        "vulkan",
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    return (
        Device::metal(burn::prelude::DeviceKind::DefaultDevice),
        "metal",
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        feature = "wgpu"
    ))]
    return (
        Device::wgpu(burn::prelude::DeviceKind::DefaultDevice),
        "wgpu",
    );
    // Both CPU backends report `cpu`: they are one accelerator as far as the
    // manifest and the daemon are concerned.
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        feature = "cpu"
    ))]
    return (Device::cpu(), "cpu");
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        feature = "flex"
    ))]
    return (Device::flex(), "cpu");
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        not(feature = "flex")
    ))]
    (Device::default(), "cpu")
}

/// The dtype the weights are cast to.
///
/// On a GPU, bf16 — the dtype the checkpoints ship in — when the device can
/// compute in it, which halves the weights and their bandwidth. f16 is the
/// next choice; it is also what the candle backend ran in, and the parity
/// test measures the two as equally faithful.
///
/// Never bf16 on Vulkan. SPIR-V's bf16 extension allows the type only in
/// conversions, dot products and cooperative matrices, never in arithmetic,
/// yet CubeCL compiles bf16 arithmetic whenever a driver reports the type.
/// That code is invalid: some drivers compute garbage from it, and NVIDIA's
/// 610.57 on an RTX 3090 segfaults in its SPIR-V compiler on the first kernel
/// that does any, the tanh GELU after the encoder's first convolution.
/// f16 transcribes there as it does on CUDA. f32 is the next fallback: exact,
/// but its weights alone are 19 GB, which a 24 GB card only runs through by
/// retrying allocations that ran out of memory.
///
/// On a CPU, f32: bf16 is slower there rather than faster.
pub fn model_dtype(device: &Device) -> DType {
    if !ON_GPU {
        return DType::F32;
    }
    let candidates: &[DType] = if cfg!(feature = "vulkan") {
        &[DType::F16]
    } else {
        &[DType::BF16, DType::F16]
    };
    candidates
        .iter()
        .copied()
        .find(|&dtype| device.supports_dtype(dtype))
        .unwrap_or(DType::F32)
}

/// Configure CubeCL: where it keeps compiled kernels, and which stream the
/// work runs on.
///
/// CubeCL compiles every kernel it meets at runtime and keeps them on disk so
/// only the first run of a build pays. Left to itself it writes under
/// `$HOME`, which the sandbox mounts read-only, so it would recompile on every
/// load.
///
/// It also gives every thread a stream of its own, and every stream memory
/// pools of its own. Transcriptions run on tokio's blocking pool, whose idle
/// threads exit after ten seconds, so requests further apart than that each
/// landed on a new thread, and so on new pools: 2.3 GB more per request,
/// measured, until the card ran out. The engine runs one request at a time
/// anyway, so one stream for everything costs nothing.
///
/// Must run before the first device is created: the configuration is frozen
/// the first time anything reads it.
pub fn configure_cubecl(cache_dir: Option<&Path>) {
    use burn::cubecl::config::cache::CacheConfig;
    use burn::cubecl::config::streaming::StreamPolicy;
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    if let Some(dir) = cache_dir {
        config.environment.path = CacheConfig::Directory(dir.to_path_buf());
    }
    config.streaming.policy = StreamPolicy::Single;
    // `false` means something already read the configuration and this call is
    // too late to matter. Nothing touches a device before `main` calls this,
    // so it is a guard rather than a case to handle.
    if !CubeClRuntimeConfig::try_set(config) {
        warn!("the CubeCL configuration was already read; it keeps its defaults");
    }
}

/// A loaded Voxtral model ready to transcribe 16 kHz mono audio.
pub struct VoxtralEngine {
    model: Voxtral,
    tokenizer: Tekkenizer,
    mel_filters: Vec<f32>,
    device_name: &'static str,
}

impl VoxtralEngine {
    /// Load the model from a directory containing `config.json`, `tekken.json`
    /// and the `*.safetensors` shards, then warm it up.
    pub fn load(model_dir: &Path, device: Option<&str>) -> Result<Self> {
        let files = resolve_files(model_dir)?;
        let config = VoxtralConfig::from_json(&std::fs::read_to_string(&files.config)?)
            .with_context(|| format!("parsing {}", files.config.display()))?;
        let (device, device_name) = select_device(device);
        let dtype = model_dtype(&device);
        info!(
            "loading Voxtral on {device_name} ({dtype:?}) from {}",
            model_dir.display()
        );
        let started = std::time::Instant::now();
        let model = Voxtral::load(&config, &files.weights, dtype, &device).map_err(Error::msg)?;
        let tokenizer = Tekkenizer::from_file(&files.tokenizer).map_err(Error::msg)?;
        info!("Voxtral mapped in {:.1?}", started.elapsed());
        let mut engine = Self {
            model,
            tokenizer,
            mel_filters: audio::mel_filters(),
            device_name,
        };
        engine.warm_up();
        Ok(engine)
    }

    /// Device label for `GET /v1/status`.
    pub fn device_label(&self) -> &'static str {
        self.device_name
    }

    /// Compile and tune the kernels a transcription runs before the first
    /// request needs them.
    ///
    /// CubeCL compiles kernels the first time it meets them and tunes each
    /// operation against its candidates per shape, so without this the first
    /// request of a load pays for all of it. A clip of silence walks the
    /// encoder, the projector, the prefill and a few decoding steps at the
    /// shapes a clip under 30 seconds — the daemon's usual request — uses.
    ///
    /// A failure is logged and swallowed: the model is loaded and usable, and
    /// refusing the load over a warm-up would turn a slow first request into
    /// no service at all.
    fn warm_up(&mut self) {
        let started = std::time::Instant::now();
        let silence = vec![0.0; CHUNK_SAMPLES];
        // Forced, since the model is right to answer silence with its
        // end-of-sequence token and a run that stops at the prefill warms no
        // decoding step.
        let result = self.decode(&silence, "en", Some(&WARM_UP_TOKENS));
        match result {
            Ok(tokens) => info!(
                "warmed up in {:.1?} ({} tokens)",
                started.elapsed(),
                tokens.len()
            ),
            Err(e) => warn!("the warm-up failed after {:.1?}: {e:#}", started.elapsed()),
        }
        self.model.release_scratch();
    }

    /// Transcribe 16 kHz mono f32 audio. (The daemon resamples upstream.)
    pub fn transcribe(
        &mut self,
        audio_data: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        if sample_rate != SAMPLE_RATE {
            warn!("Voxtral expects {SAMPLE_RATE}Hz; got {sample_rate}Hz (daemon should resample)");
        }
        // Voxtral conditions on a `lang:<code>` prompt token and has no
        // auto-detect mode, so the reserved `auto` (and an omitted language)
        // fall back to the model's primary (English). The daemon only sends
        // codes the model supports.
        let lang_code = match language {
            Some("auto") | None => "en",
            Some(code) => code,
        };
        let started = std::time::Instant::now();
        let tokens = self.decode(audio_data, lang_code, None)?;
        let text = self
            .tokenizer
            .decode(&tokens, tekken::SpecialTokenPolicy::Ignore)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))?;
        info!(
            "transcribed {:.1}s of audio into {} tokens in {:.1?}",
            seconds(audio_data.len()),
            tokens.len(),
            started.elapsed()
        );
        post_process_transcription(&text)
    }

    /// Hand the last transcription's working memory back to the driver.
    ///
    /// The pools keep the pages a request's activations lived in, so an idle
    /// backend would otherwise hold its largest request's working memory —
    /// over 2 GB on top of the weights — for as long as it stays loaded. The
    /// release takes about 70 ms and the next request about 50 ms more to
    /// allocate again, so the caller runs this after replying, not before.
    pub fn release_memory(&self) {
        self.model.release_scratch();
    }

    /// Audio to generated tokens, the model's own unless `forced`.
    fn decode(
        &self,
        audio_data: &[f32],
        lang_code: &str,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        let padded = audio::pad_to_chunk(audio_data, CHUNK_SAMPLES);
        let features = audio::extract_features(&padded, &self.mel_filters);
        let config = self.model.config();
        let input_ids = prompt(
            &self.tokenizer,
            features.chunks * config.audio_tokens_per_chunk(),
            config.audio_token_id,
            lang_code,
        )?;
        let audio = self.model.encode_audio(&features, &mut Taps::off());
        self.model
            .generate(&input_ids, audio, MAX_NEW_TOKENS, forced, &mut Taps::off())
            .map_err(|e| anyhow::anyhow!("Failed to generate tokens: {e}"))
    }
}

#[allow(clippy::cast_precision_loss)]
fn seconds(samples: usize) -> f64 {
    samples as f64 / f64::from(SAMPLE_RATE)
}

/// `<s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:<code>[TRANSCRIBE]`.
///
/// The `lang:<code>` hint is tokenized: `lang:en` reproduces the model's
/// original `[9909, 1058, 1262]`.
fn prompt(
    tokenizer: &Tekkenizer,
    audio_tokens: usize,
    audio_token_id: u32,
    lang_code: &str,
) -> Result<Vec<u32>> {
    let lang = tokenizer
        .encode(&format!("lang:{lang_code}"), false, false)
        .map_err(|e| anyhow::anyhow!("encode lang prompt: {e}"))?;
    let mut tokens = Vec::with_capacity(PROMPT_OPEN.len() + audio_tokens + lang.len() + 2);
    tokens.extend(PROMPT_OPEN);
    tokens.extend(std::iter::repeat_n(audio_token_id, audio_tokens));
    tokens.push(INST_CLOSE);
    tokens.extend(lang);
    tokens.push(TRANSCRIBE);
    Ok(tokens)
}

#[derive(Debug)]
struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: Vec<PathBuf>,
}

fn resolve_files(dir: &Path) -> Result<ModelFiles> {
    let config = dir.join("config.json");
    anyhow::ensure!(
        config.exists(),
        "config.json not found in {}",
        dir.display()
    );
    let tokenizer = dir.join("tekken.json");
    anyhow::ensure!(
        tokenizer.exists(),
        "tekken.json not found in {}",
        dir.display()
    );
    let mut weights = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let p = entry?.path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            weights.push(p);
        }
    }
    anyhow::ensure!(!weights.is_empty(), "no .safetensors in {}", dir.display());
    weights.sort();
    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

/// Clean up Voxtral output formatting artifacts. (Ported verbatim.)
fn post_process_transcription(text: &str) -> Result<String> {
    let mut cleaned = text.trim().to_string();
    if cleaned.starts_with("\"'") || cleaned.starts_with("'\"") {
        cleaned = cleaned
            .trim_start_matches("\"'")
            .trim_start_matches("'\"")
            .trim()
            .to_string();
    }
    if cleaned.starts_with('\'') {
        cleaned = cleaned[1..].trim().to_string();
    }
    cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    cleaned = cleaned.replace(" \"' ", " ").replace(" '\" ", " ");
    if cleaned == "." || cleaned.trim().is_empty() {
        anyhow::bail!("mel feature generation produced invalid output");
    }
    cleaned = cleaned
        .trim_end_matches('\'')
        .trim_end_matches('"')
        .to_string();
    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tekken_path() -> Option<String> {
        std::env::var("SUPER_STT_TEST_TEKKEN")
            .ok()
            .or_else(|| {
                std::env::var("SUPER_STT_BACKEND_DIR")
                    .ok()
                    .map(|d| format!("{d}/models/voxtral-mini-3b-2507/tekken.json"))
            })
            .filter(|p| Path::new(p).exists())
    }

    #[test]
    fn post_process_trims_and_collapses_whitespace() {
        assert_eq!(
            post_process_transcription("  hello   world  ").unwrap(),
            "hello world"
        );
    }

    /// Guards the `lang:<code>` prompt tokenization: `lang:en` must reproduce
    /// the model's original hardcoded ids `[9909, 1058, 1262]`, or the
    /// transcription prompt would silently break on a tokenizer change.
    /// Tokenizer-only (no model/GPU); self-skips unless a `tekken.json` is
    /// provisioned via `SUPER_STT_TEST_TEKKEN` or `SUPER_STT_BACKEND_DIR`.
    #[test]
    fn lang_prompt_encoding_is_stable() {
        let Some(path) = tekken_path() else {
            return; // no tokenizer provisioned
        };
        let t = Tekkenizer::from_file(&path).expect("load tekken.json");
        assert_eq!(
            t.encode("lang:en", false, false).expect("encode lang:en"),
            vec![9909, 1058, 1262],
            "lang:en tokenization changed — the Voxtral prompt would break",
        );
    }

    /// The whole prompt around the audio, against the ids the candle backend
    /// built. Self-skips without a tokenizer, as above.
    #[test]
    fn prompt_wraps_the_audio_placeholders() {
        let Some(path) = tekken_path() else {
            return;
        };
        let t = Tekkenizer::from_file(&path).expect("load tekken.json");
        let ids = prompt(&t, 750, 24, "en").unwrap();
        assert_eq!(&ids[..3], &[1, 3, 25]);
        assert!(ids[3..753].iter().all(|&id| id == 24));
        assert_eq!(&ids[753..], &[4, 9909, 1058, 1262, 34]);
    }

    #[test]
    fn post_process_strips_leading_quote_apostrophe() {
        assert_eq!(
            post_process_transcription("\"'hello world").unwrap(),
            "hello world"
        );
        assert_eq!(
            post_process_transcription("'\"hello world").unwrap(),
            "hello world"
        );
    }

    #[test]
    fn post_process_strips_leading_apostrophe() {
        assert_eq!(post_process_transcription("'hello").unwrap(), "hello");
    }

    #[test]
    fn post_process_strips_trailing_quote_and_apostrophe() {
        assert_eq!(post_process_transcription("hello\"").unwrap(), "hello");
        assert_eq!(post_process_transcription("hello'").unwrap(), "hello");
    }

    #[test]
    fn post_process_rejects_empty_and_bare_dot() {
        assert!(post_process_transcription("").is_err());
        assert!(post_process_transcription("    ").is_err());
        assert!(post_process_transcription(".").is_err());
    }

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"x").unwrap();
    }

    #[test]
    fn resolve_files_requires_config() {
        let d = tempfile::tempdir().unwrap();
        let err = resolve_files(d.path()).unwrap_err().to_string();
        assert!(err.contains("config.json not found"), "{err}");
    }

    #[test]
    fn resolve_files_requires_tokenizer() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "config.json");
        let err = resolve_files(d.path()).unwrap_err().to_string();
        assert!(err.contains("tekken.json not found"), "{err}");
    }

    #[test]
    fn resolve_files_requires_safetensors() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "config.json");
        touch(d.path(), "tekken.json");
        let err = resolve_files(d.path()).unwrap_err().to_string();
        assert!(err.contains("no .safetensors"), "{err}");
    }

    #[test]
    fn resolve_files_sorts_weight_shards() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "config.json");
        touch(d.path(), "tekken.json");
        touch(d.path(), "model-00002.safetensors");
        touch(d.path(), "model-00001.safetensors");
        let files = resolve_files(d.path()).unwrap();
        let names: Vec<_> = files
            .weights
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            ["model-00001.safetensors", "model-00002.safetensors"]
        );
    }

    #[test]
    fn the_reported_accelerator_matches_the_compiled_backend() {
        let (_, name) = select_device(None);
        // `flex` and `cpu` are two CPU backends and both report "cpu".
        assert_eq!(name, BUILT_FOR);
        assert_eq!(ON_GPU, BUILT_FOR != "cpu");
        let (device, _) = select_device(None);
        assert_eq!(model_dtype(&device) == DType::F32, !ON_GPU);
    }

    #[test]
    fn a_mismatched_device_request_still_uses_the_compiled_backend() {
        let other = if BUILT_FOR == "cpu" { "cuda" } else { "cpu" };
        assert_eq!(select_device(Some(other)).1, BUILT_FOR);
        assert_eq!(select_device(Some("")).1, BUILT_FOR);
    }
}
