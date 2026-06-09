// SPDX-License-Identifier: GPL-3.0-only
//! Mock-daemon protocol test against the real binary with NO model provisioned.
//! Verifies the `/v1` HTTP surface, the pre-load `starting` state, and a
//! graceful load-error path. CPU-only — runs in CI.

mod common;
use common::Backend;
use std::time::Duration;

#[tokio::test]
async fn ping_and_starting_state() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Backend::spawn(dir.path()).await;

    let (code, body) = backend
        .request("GET", "/v1/ping", Vec::new())
        .await
        .unwrap();
    assert_eq!(code, 200);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["message"], "pong");

    let s = backend.status().await;
    assert_eq!(s["state"], "starting", "no load yet → starting");
}

#[tokio::test]
async fn load_without_weights_reports_error() {
    let dir = tempfile::tempdir().unwrap(); // no models/ subdir
    let backend = Backend::spawn(dir.path()).await;

    assert_eq!(backend.load("voxtral-mini", "cpu").await, 202);

    // Weights dir is missing → the engine load fails → state becomes `error`.
    let s = backend
        .wait_for_state(&["error"], Duration::from_secs(30))
        .await;
    assert_eq!(s["state"], "error");
    assert!(
        s.get("reason").and_then(|r| r.as_str()).is_some(),
        "error state must carry a reason: {s}"
    );
}
