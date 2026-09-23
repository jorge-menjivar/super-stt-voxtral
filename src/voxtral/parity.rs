// SPDX-License-Identifier: GPL-3.0-only
//! Layer-by-layer comparison with candle, the implementation this backend ran
//! on before the port.
//!
//! `parity/` dumps every layer's output of candle's Voxtral over one clip; this
//! runs the port over the same inputs and compares each against it. Gated,
//! since it needs the weights and the dump:
//!
//! ```sh
//! just parity                       # dumps with candle, then runs this
//! SUPER_STT_PARITY_REF=<dump.safetensors> SUPER_STT_BACKEND_DIR=<dir> \
//!   cargo test --release parity -- --nocapture
//! ```
//!
//! `SUPER_STT_PARITY_DTYPE` picks the port's dtype, `f32` by default. What is
//! held depends on it:
//!
//! - f32 on the CPU is the same arithmetic as candle's in a different order,
//!   and every layer is held within 1e-3 of it (measured: 7e-4 at worst, deep
//!   in the decoder);
//! - f32 on a GPU is not quite f32: the first convolution already sits 2e-4
//!   from candle's, about where TF32 tensor cores would put it, and the error
//!   compounds from there — so it is held to the baseline below instead: no
//!   layer may drift further than candle's own f16 does;
//! - in f16 and bf16 the table is the measurement.
//!
//! The greedy tokens must match candle's in every case.
//!
//! What a narrower type *should* cost is what it costs candle, so
//! `SUPER_STT_PARITY_BASELINE` takes a second dump — candle over the same clip
//! in f16, which is how the candle backend shipped — and adds its distance
//! from the f32 reference to every row. A port drifting further than that is
//! doing something candle does not. (The port in f16 lands on or under that
//! column on every row; in bf16 it sits 1.2–3x above it, which is the three
//! mantissa bits bf16 gives up.)
//!
//! The features are compared on their own first; the model is then fed
//! candle's, so the comparison starts from identical input. From there each
//! layer reads the port's own output of the layer before, so the error at a
//! layer is what it adds plus everything carried in — the table shows how it
//! grows with depth. The decoding steps are teacher-forced with candle's
//! tokens, so every step's logits are compared even past a near-tie.

use std::path::{Path, PathBuf};

use burn::tensor::DType;
use safetensors::{Dtype, SafeTensors};

use crate::inference::{MAX_NEW_TOKENS, select_device};
use crate::voxtral::audio::{self, Features};
use crate::voxtral::config::VoxtralConfig;
use crate::voxtral::model::{Taps, Voxtral};

/// How far one tensor is from another.
#[derive(Debug, Clone, Copy)]
struct Diff {
    max_abs: f64,
    /// `‖ours − theirs‖ / ‖theirs‖`.
    rel_l2: f64,
    cosine: f64,
    /// `max |theirs|`, for reading `max_abs` against.
    scale: f64,
}

fn diff(ours: &[f32], theirs: &[f32]) -> Diff {
    assert_eq!(ours.len(), theirs.len());
    let (mut max_abs, mut sq_err, mut sq_ref, mut sq_ours, mut dot, mut scale) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(theirs) {
        let (a, b) = (f64::from(a), f64::from(b));
        max_abs = max_abs.max((a - b).abs());
        sq_err += (a - b) * (a - b);
        sq_ref += b * b;
        sq_ours += a * a;
        dot += a * b;
        scale = scale.max(b.abs());
    }
    Diff {
        max_abs,
        rel_l2: (sq_err / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        cosine: dot / (sq_ref.sqrt() * sq_ours.sqrt()).max(f64::MIN_POSITIVE),
        scale,
    }
}

fn f32s(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::F32, "{name} is not f32");
    let values = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), values)
}

fn u32s(st: &SafeTensors<'_>, name: &str) -> Vec<u32> {
    let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::U32, "{name} is not u32");
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("SUPER_STT_PARITY_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("SUPER_STT_BACKEND_DIR")
                .map(|d| Path::new(&d).join("models/voxtral-mini-3b-2507"))
        })
}

#[test]
#[allow(clippy::too_many_lines)]
fn layers_match_candle() {
    let Some(reference) = std::env::var_os("SUPER_STT_PARITY_REF") else {
        return; // no candle dump provisioned
    };
    let model_dir = model_dir().expect("SUPER_STT_BACKEND_DIR or SUPER_STT_PARITY_MODEL_DIR");
    let dtype = match std::env::var("SUPER_STT_PARITY_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        Ok("f32") | Err(_) => DType::F32,
        Ok(other) => panic!("SUPER_STT_PARITY_DTYPE takes f32, bf16 or f16, not {other}"),
    };
    let strict = dtype == DType::F32;
    let bytes = std::fs::read(&reference).expect("reading the candle dump");
    let st = SafeTensors::deserialize(&bytes).expect("parsing the candle dump");
    let baseline_bytes = std::env::var_os("SUPER_STT_PARITY_BASELINE")
        .map(|path| std::fs::read(path).expect("reading the baseline dump"));
    let baseline = baseline_bytes
        .as_deref()
        .map(|bytes| SafeTensors::deserialize(bytes).expect("parsing the baseline dump"));

    // The features, from the same samples candle was given.
    let (_, samples) = f32s(&st, "audio");
    let ours = audio::extract_features(&samples, &audio::mel_filters());
    let (mel_shape, mel) = f32s(&st, "mel");
    assert_eq!(ours.chunks, mel_shape[0], "chunk count");
    let features = diff(&ours.data, &mel);
    eprintln!(
        "features: {} chunks, max |Δ| {:.3e}, bit-identical: {}",
        ours.chunks,
        features.max_abs,
        ours.data == mel
    );
    assert!(
        ours.data == mel,
        "the features differ from candle's by up to {}",
        features.max_abs
    );

    // The model, from candle's features, prompt and tokens.
    let config = VoxtralConfig::from_json(
        &std::fs::read_to_string(model_dir.join("config.json")).expect("config.json"),
    )
    .expect("parsing config.json");
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)
        .expect("model dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    let (device, device_name) = select_device(None);
    let started = std::time::Instant::now();
    let model = Voxtral::load(&config, &shards, dtype, &device).expect("loading the weights");
    eprintln!(
        "loaded on {device_name} in {dtype:?} in {:.1?}",
        started.elapsed()
    );

    let input_ids = u32s(&st, "input_ids");
    let tokens = u32s(&st, "tokens");
    let mut taps = Taps::on();
    let started = std::time::Instant::now();
    let audio = model.encode_audio(
        &Features {
            data: mel,
            chunks: mel_shape[0],
        },
        &mut taps,
    );
    let ours = model
        .generate(&input_ids, audio, MAX_NEW_TOKENS, Some(&tokens), &mut taps)
        .expect("generating");
    eprintln!("ran in {:.1?}", started.elapsed());

    eprintln!(
        "\n{:<14} {:>18} {:>11} {:>11} {:>11} {:>13} {:>14}",
        "tap", "shape", "max |Δ|", "max |ref|", "rel L2", "1 − cos", "baseline rel"
    );
    let mut worst_rel = 0f64;
    let mut compared = 0;
    let mut past_baseline = Vec::new();
    for (name, values) in taps.into_records() {
        let (shape, theirs) = f32s(&st, &name);
        assert_eq!(values.len(), theirs.len(), "{name}: {shape:?}");
        let d = diff(&values, &theirs);
        let base_rel = baseline
            .as_ref()
            .map(|b| diff(&f32s(b, &name).1, &theirs).rel_l2);
        if base_rel.is_some_and(|base| d.rel_l2 > base) {
            past_baseline.push(name.clone());
        }
        let base = base_rel.map_or_else(String::new, |base| format!("{base:.3e}"));
        eprintln!(
            "{:<14} {:>18} {:>11.3e} {:>11.3e} {:>11.3e} {:>13.3e} {:>14}",
            name,
            format!("{shape:?}"),
            d.max_abs,
            d.scale,
            d.rel_l2,
            1.0 - d.cosine,
            base
        );
        worst_rel = worst_rel.max(d.rel_l2);
        compared += 1;
    }
    let agree = ours.iter().zip(&tokens).filter(|(a, b)| a == b).count();
    eprintln!(
        "\n{compared} taps; worst rel L2 {worst_rel:.3e}; greedy tokens agree at {agree} of {} steps",
        tokens.len()
    );
    // mel, 3 convolution/embedding taps, the encoder layers and norm, the
    // projector, the spliced embeddings, the decoder layers and norm, and the
    // logits of every step.
    let expected = 1
        + 3
        + config.audio_config.num_hidden_layers
        + 1
        + 1
        + 1
        + config.text_config.num_hidden_layers
        + 1
        + tokens.len();
    assert_eq!(compared, expected, "a tap is missing from one side");
    assert_eq!(ours, tokens, "greedy decoding diverged from candle's");
    if strict && device_name == "cpu" {
        assert!(
            worst_rel < 1e-3,
            "in f32 every layer should be within 1e-3 of candle's, the worst is {worst_rel:.3e}"
        );
    }
    if strict {
        assert!(
            past_baseline.is_empty(),
            "in f32 no layer should drift further than candle's baseline, these do: {past_baseline:?}"
        );
    }
}
