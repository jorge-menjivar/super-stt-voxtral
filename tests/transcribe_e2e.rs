// SPDX-License-Identifier: GPL-3.0-only
//! Gated real-model E2E (GPU). Provision `models/voxtral-mini` under
//! SUPER_STT_BACKEND_DIR, then run with `--features cuda`:
//!   SUPER_STT_TEST_VOXTRAL=1 SUPER_STT_BACKEND_DIR=<dir> \
//!   SUPER_STT_TEST_AUDIO=<wav> cargo test --features cuda --test transcribe_e2e -- --nocapture

#![allow(clippy::doc_markdown)] // env var names in shell-command doc comment

mod common;
use common::Backend;
use std::path::PathBuf;
use std::time::Duration;

fn read_wav_mono_f32(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.expect("sample")) / f32::from(i16::MAX))
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("sample"))
            .collect(),
    };
    (samples, spec.sample_rate)
}

#[tokio::test]
async fn transcribe_real_model() {
    if std::env::var("SUPER_STT_TEST_VOXTRAL").is_err() {
        return; // self-skip unless explicitly enabled
    }
    let backend_dir = PathBuf::from(
        std::env::var("SUPER_STT_BACKEND_DIR")
            .expect("SUPER_STT_BACKEND_DIR (must contain models/voxtral-mini)"),
    );
    assert!(
        backend_dir.join("models/voxtral-mini").exists(),
        "provision models/voxtral-mini under SUPER_STT_BACKEND_DIR first"
    );
    let audio_path = std::env::var("SUPER_STT_TEST_AUDIO").expect("SUPER_STT_TEST_AUDIO");

    let backend = Backend::spawn(&backend_dir).await;
    assert_eq!(backend.load("voxtral-mini", "").await, 202);
    backend
        .wait_for_state(&["ready"], Duration::from_mins(10))
        .await;

    let (samples, sr) = read_wav_mono_f32(&audio_path);
    let (code, body) = backend.transcribe(&samples, sr).await;
    assert_eq!(code, 200, "transcribe failed: {body}");
    let text = body["transcription"].as_str().unwrap_or("");
    println!("=== VOXTRAL TRANSCRIPTION ===\n{text}\n=============================");
    assert!(!text.trim().is_empty(), "expected non-empty transcription");
    if let Ok(expect) = std::env::var("SUPER_STT_TEST_EXPECT") {
        assert!(
            text.to_lowercase().contains(&expect.to_lowercase()),
            "transcription {text:?} did not contain {expect:?}"
        );
    }
}
