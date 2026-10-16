// SPDX-License-Identifier: GPL-3.0-only
//! The engine the `/v1` handlers drive: which device to run on, where the
//! files are, how a request becomes a prompt, and how the tokens that come
//! back become a transcript. The model itself is in [`crate::voxtral`].

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Error, Result};
use burn::prelude::Device;
use burn::tensor::DType;
use log::{info, warn};
use tekken::Tekkenizer;

use crate::progress::{Measure, Phase, Report, Step, Tracker};
use crate::voxtral::audio::{self, CHUNK_SAMPLES, SAMPLE_RATE};
use crate::voxtral::config::VoxtralConfig;
use crate::voxtral::model::{Taps, Voxtral};

/// Longest transcript, in tokens. It also sizes the key/value caches, see
/// [`Voxtral::generate`].
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

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
)))]
compile_error!("build with one accelerator: `cuda`, `rocm`, `vulkan` or `metal`");

/// The accelerator this build was compiled for, as the manifest names it.
///
/// One build serves one accelerator: Burn's backends are cargo features, and
/// the asset that carries this binary declares the matching `accel`. Every one
/// is a GPU; the models are too big for a CPU build to be worth shipping.
const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else {
    "metal"
};

/// The entries a first load's warm-up writes to the kernel cache, which is
/// what its progress is measured against: tuning results and compiled
/// kernels, each with a record beside it, counted from an empty cache for
/// each accelerator (see `docs/port-notes.md`); Metal uses Vulkan's count.
/// Only the pace of the bar rides on it: past the estimate it slows down short
/// of the end rather than stopping, see `progress::estimate`.
const WARM_UP_CACHE_ENTRIES: u64 = if cfg!(feature = "cuda") {
    1166
} else if cfg!(feature = "rocm") {
    351
} else {
    553
};

/// Encoder and decoder layers the kernel build keeps, see
/// [`VoxtralEngine::build_kernels`]: two, so that what runs between one layer
/// and the next is there too.
const BUILD_LAYERS: usize = 2;

/// Where CubeCL keeps this backend's kernels, when the daemon grants a place.
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

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
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan")))]
    (
        Device::metal(burn::prelude::DeviceKind::DefaultDevice),
        "metal",
    )
}

/// The dtype the weights are cast to: f16 on every accelerator, and f32 on a
/// device without it.
///
/// f16 halves the weights and their bandwidth, and of the two 16-bit types it
/// keeps more mantissa bits; what bf16 buys instead is range, which Voxtral's
/// activations do not need. bf16, the dtype the checkpoints ship in, is also
/// not one every CubeCL backend compiles, whatever `supports_dtype` reports
/// (see `docs/port-notes.md`). f32 is the last resort: exact, but twice the
/// memory.
pub fn model_dtype(device: &Device) -> DType {
    if device.supports_dtype(DType::F16) {
        DType::F16
    } else {
        DType::F32
    }
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
/// land on a new thread, and so on new pools, until the card runs out. The
/// engine runs one request at a time anyway, so one stream for everything
/// costs nothing.
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
        let _ = CACHE_DIR.set(dir.to_path_buf());
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
    /// and the `*.safetensors` shards, then warm it up, reporting how far it
    /// has got into `report` as it goes.
    pub fn load(
        model_dir: &Path,
        device: Option<&str>,
        report: &(dyn Fn(Report) + Sync),
    ) -> Result<Self> {
        let files = resolve_files(model_dir)?;
        let config = VoxtralConfig::from_json(&std::fs::read_to_string(&files.config)?)
            .with_context(|| format!("parsing {}", files.config.display()))?;
        let marker = warm_marker_for(model_dir);
        let phase = if marker.as_ref().is_some_and(|m| m.exists()) {
            Phase::Loading
        } else {
            Phase::InitialSetup
        };
        info!("loading Voxtral from {} ({phase:?})", model_dir.display());
        let tracker = Tracker::new(phase, report);
        let engine = std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            // Stops the sampler however this ends, a panic included: the
            // scope waits for it before it lets a panic through.
            let _finish = Finish(&tracker);
            Self::load_tracked(&files, &config, device, phase, &tracker)
        })?;
        mark_warm(marker.as_deref());
        Ok(engine)
    }

    /// Compile and tune the kernels a load of the model in `model_dir` needs,
    /// from its `config.json` and `tekken.json` alone, so that they can be
    /// built while the weights are still on their way.
    ///
    /// The kernels depend on the shapes, dtypes and layouts the model runs at,
    /// not on its weights, so this runs the warm-up on a model whose weights
    /// are zeros. Every encoder layer runs the same kernels at the same
    /// shapes, and so does every decoder layer, so it keeps [`BUILD_LAYERS`]
    /// of each. A later load of the model then finds them cached and is not an
    /// initial setup. Its memory is handed back before this returns, so the
    /// load that follows has the device to itself.
    pub fn build_kernels(
        model_dir: &Path,
        device: Option<&str>,
        report: &(dyn Fn(Report) + Sync),
    ) -> Result<()> {
        let config_path = model_dir.join("config.json");
        let mut config = VoxtralConfig::from_json(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        config.audio_config.num_hidden_layers =
            config.audio_config.num_hidden_layers.min(BUILD_LAYERS);
        config.text_config.num_hidden_layers =
            config.text_config.num_hidden_layers.min(BUILD_LAYERS);
        let tokenizer = Tekkenizer::from_file(model_dir.join("tekken.json")).map_err(Error::msg)?;
        let (device, device_name) = select_device(device);
        let dtype = model_dtype(&device);
        info!(
            "building Voxtral's kernels from {} on {device_name}, in {dtype:?}",
            model_dir.display()
        );
        let tracker = Tracker::new(Phase::InitialSetup, report);
        let built = std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            let _finish = Finish(&tracker);
            tracker.enter(
                Step::BuildingKernels,
                Measure::cache_entries(WARM_UP_CACHE_ENTRIES),
            );
            let mut engine = Self {
                model: Voxtral::zeroed(&config, dtype, &device),
                tokenizer,
                mel_filters: audio::mel_filters(),
                device_name,
            };
            engine.warm_up()
        });
        device.memory_cleanup();
        if let Err(err) = device.sync() {
            warn!("syncing the device after building the kernels failed: {err}");
        }
        built?;
        mark_warm(warm_marker_for(model_dir).as_deref());
        Ok(())
    }

    /// Whether a load of the model in `model_dir` would find its kernels
    /// built by this build.
    pub fn kernels_built(model_dir: &Path) -> bool {
        warm_marker_for(model_dir).is_some_and(|marker| marker.exists())
    }

    /// [`Self::load`] under its tracker.
    fn load_tracked(
        files: &ModelFiles,
        config: &VoxtralConfig,
        device: Option<&str>,
        phase: Phase,
        tracker: &Tracker<'_>,
    ) -> Result<Self> {
        let (device, device_name) = select_device(device);
        let dtype = model_dtype(&device);
        info!("on {device_name}, in {dtype:?}");
        let started = std::time::Instant::now();
        let total = files.weights.iter().try_fold(0, |sum, shard| {
            std::fs::metadata(shard).map(|m| sum + m.len())
        })?;
        let read = Arc::new(AtomicU64::new(0));
        tracker.enter(
            Step::LoadingWeights,
            Measure::Bytes {
                read: Arc::clone(&read),
                total,
            },
        );
        let model =
            Voxtral::load(config, &files.weights, dtype, &device, &read).map_err(Error::msg)?;
        let tokenizer = Tekkenizer::from_file(&files.tokenizer).map_err(Error::msg)?;
        info!("Voxtral mapped in {:.1?}", started.elapsed());
        let mut engine = Self {
            model,
            tokenizer,
            mel_filters: audio::mel_filters(),
            device_name,
        };
        let step = match phase {
            Phase::InitialSetup => Step::BuildingKernels,
            Phase::Loading => Step::WarmingUp,
        };
        tracker.enter(step, Measure::cache_entries(WARM_UP_CACHE_ENTRIES));
        engine.warm_up()?;
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
    /// A failure fails the load. The warm-up runs what every transcription
    /// runs, so a kernel that will not compile or a card that runs out of
    /// memory here fails each request after it too; better the daemon hears
    /// it now, with the reason, than a backend that says `ready` and never
    /// transcribes.
    fn warm_up(&mut self) -> Result<()> {
        let started = std::time::Instant::now();
        let silence = vec![0.0; CHUNK_SAMPLES];
        // Forced, since the model is right to answer silence with its
        // end-of-sequence token and a run that stops at the prefill warms no
        // decoding step.
        let result = self.decode(&silence, "en", Some(&WARM_UP_TOKENS));
        self.model.release_scratch();
        let tokens = result
            .with_context(|| format!("the warm-up failed after {:.1?}", started.elapsed()))?;
        info!(
            "warmed up in {:.1?} ({} tokens)",
            started.elapsed(),
            tokens.len()
        );
        Ok(())
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
    /// backend would otherwise hold its largest request's working memory on
    /// top of the weights for as long as it stays loaded. Releasing it, and
    /// the next request allocating it again, take a little time, so the
    /// caller runs this after replying, not before.
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

/// Where a load or a kernel build records that the model in `model_dir`
/// finished a warm-up with this build, so that the next load of it is not an
/// initial setup. It sits beside the
/// kernels it vouches for, so clearing the cache clears it too. Without a
/// cache directory there is none, and every load is an initial setup, since
/// every load compiles everything.
///
/// Named by the binary's build ID, which is what CubeCL keys its compiled
/// kernels on: any rebuild, a release that keeps its version included,
/// recompiles all of them, and should say so. Without a build ID the crate
/// version stands in; every release asset has one.
fn warm_marker_for(model_dir: &Path) -> Option<PathBuf> {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(warm_marker)
}

/// [`warm_marker_for`] by the model's name.
fn warm_marker(model: &str) -> Option<PathBuf> {
    let build = buildid::build_id().map_or_else(
        || env!("CARGO_PKG_VERSION").to_string(),
        |id| {
            id.iter().fold(String::new(), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
        },
    );
    CACHE_DIR.get().map(|dir| {
        dir.join("voxtral-warm")
            .join(format!("{model}-{BUILT_FOR}-{build}"))
    })
}

/// Writes `marker`, if there is one, recording a finished warm-up.
fn mark_warm(marker: Option<&Path>) {
    let Some(marker) = marker else {
        return;
    };
    let written = marker
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(marker, b""));
    if let Err(e) = written {
        warn!("could not mark {} warm: {e}", marker.display());
    }
}

/// Stops a [`Tracker`] when dropped.
struct Finish<'a, 'b>(&'a Tracker<'b>);

impl Drop for Finish<'_, '_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

#[allow(clippy::cast_precision_loss)]
fn seconds(samples: usize) -> f64 {
    samples as f64 / f64::from(SAMPLE_RATE)
}

/// `<s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:<code>[TRANSCRIBE]`.
///
/// The `lang:<code>` hint is tokenized: `lang:en` encodes to
/// `[9909, 1058, 1262]`.
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

/// Clean up Voxtral output formatting artifacts.
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

    /// Guards the `lang:<code>` prompt tokenization: `lang:en` must encode to
    /// `[9909, 1058, 1262]`, or the transcription prompt would silently break
    /// on a tokenizer change.
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

    /// The whole prompt around the audio, id for id. Self-skips without a
    /// tokenizer, as above.
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

    // Naming the device creates nothing, so these run without a GPU; asking
    // it anything, the dtypes it supports included, would not.
    #[test]
    fn the_reported_accelerator_matches_the_compiled_backend() {
        assert_eq!(select_device(None).1, BUILT_FOR);
    }

    #[test]
    fn a_mismatched_device_request_still_uses_the_compiled_backend() {
        let other = if BUILT_FOR == "cuda" { "rocm" } else { "cuda" };
        assert_eq!(select_device(Some(other)).1, BUILT_FOR);
        assert_eq!(select_device(Some("")).1, BUILT_FOR);
    }
}
