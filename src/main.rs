// SPDX-License-Identifier: GPL-3.0-only
//! Voxtral subprocess backend: serves the Super STT `/v1` contract over a
//! pathname Unix socket (`SUPER_STT_BACKEND_SOCKET`), loading the model from
//! `SUPER_STT_BACKEND_DIR/models/<name>`. Self-contained — no super-stt deps.

// doc lint trips on prose like "candle"/"super-stt".
#![allow(clippy::doc_markdown)]

mod inference;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::UnixListener;

use inference::VoxtralEngine;

#[derive(Clone, Copy)]
enum LoadState {
    Starting,
    Loading,
    Ready,
    Error,
}

impl LoadState {
    fn as_str(self) -> &'static str {
        match self {
            LoadState::Starting => "starting",
            LoadState::Loading => "loading",
            LoadState::Ready => "ready",
            LoadState::Error => "error",
        }
    }
}

struct Status {
    state: LoadState,
    model: Option<String>,
    device: Option<String>,
    reason: Option<String>,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: LoadState::Starting,
            model: None,
            device: None,
            reason: None,
        }
    }
}

struct AppState {
    backend_dir: PathBuf,
    status: Mutex<Status>,
    engine: Mutex<Option<VoxtralEngine>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let socket = std::env::var("SUPER_STT_BACKEND_SOCKET")
        .context("SUPER_STT_BACKEND_SOCKET must be set")?;
    let backend_dir =
        std::env::var("SUPER_STT_BACKEND_DIR").context("SUPER_STT_BACKEND_DIR must be set")?;

    let state = Arc::new(AppState {
        backend_dir: PathBuf::from(backend_dir),
        status: Mutex::new(Status::default()),
        engine: Mutex::new(None),
    });

    let app = router(state);

    if let Some(parent) = std::path::Path::new(&socket).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).with_context(|| format!("bind {socket}"))?;
    log::info!("voxtral backend serving /v1 on {socket}");

    loop {
        let (stream, _) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = TowerToHyperService::new(app);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}

/// Build the `/v1` router. Extracted from `main` so handlers can be exercised
/// in-process by the tests below without spawning the binary.
fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(get_status))
        .route("/v1/load", post(load))
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/cancel", post(cancel))
        // Audio payloads (f32 arrays as JSON) easily exceed the 2 MB default.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

async fn get_status(State(s): State<Arc<AppState>>) -> Json<Value> {
    let st = s.status.lock().unwrap();
    let mut out = json!({ "status": "success", "state": st.state.as_str() });
    if let Some(m) = &st.model {
        out["model"] = json!({ "name": m });
    }
    if let Some(d) = &st.device {
        out["device"] = json!(d);
    }
    if let Some(r) = &st.reason {
        out["reason"] = json!(r);
    }
    Json(out)
}

#[derive(Deserialize)]
struct LoadReq {
    name: String,
    #[serde(default)]
    device: Option<String>,
}

async fn load(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    {
        let mut st = s.status.lock().unwrap();
        st.state = LoadState::Loading;
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
    }
    let dir = s.backend_dir.join("models").join(&req.name);
    let force_cpu = req.device.as_deref() == Some("cpu");
    let s2 = Arc::clone(&s);
    tokio::spawn(async move {
        let res = tokio::task::spawn_blocking(move || VoxtralEngine::load(&dir, force_cpu)).await;
        match res {
            Ok(Ok(engine)) => {
                let label = engine.device_label().to_string();
                *s2.engine.lock().unwrap() = Some(engine);
                let mut st = s2.status.lock().unwrap();
                st.device = Some(label);
                st.state = LoadState::Ready;
                log::info!("model loaded; ready");
            }
            Ok(Err(e)) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("{e:#}"));
                log::error!("model load failed: {e:#}");
            }
            Err(e) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("load task panicked: {e}"));
            }
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
}

#[derive(Deserialize)]
struct TranscribeReq {
    audio_data: Vec<f32>,
    #[serde(default)]
    sample_rate: Option<u32>,
}

async fn transcribe(
    State(s): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(req): Json<TranscribeReq>,
) -> (StatusCode, Json<Value>) {
    if !matches!(s.status.lock().unwrap().state, LoadState::Ready) {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "status": "error", "message": "not_ready" })),
        );
    }
    // Contract: empty `audio_data` is a client error, not an inference failure
    // (docs/protocol/backend/contract.md → 400 invalid_audio). Guarding here also
    // keeps an empty buffer out of the engine's chunk-padding (0 → zero chunks).
    if req.audio_data.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_audio" })),
        );
    }
    let sample_rate = req.sample_rate.unwrap_or(16000);
    let audio = req.audio_data;
    let s2 = Arc::clone(&s);
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = s2.engine.lock().unwrap();
        let engine = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("engine not loaded"))?;
        engine.transcribe(&audio, sample_rate)
    })
    .await;
    match result {
        Ok(Ok(text)) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "transcription": text })),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "inference_failed", "detail": format!("{e:#}") }),
            ),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "inference_panicked", "detail": format!("{e}") }),
            ),
        ),
    }
}

async fn cancel() -> Json<Value> {
    Json(json!({ "status": "success", "message": "Cancelled" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            backend_dir: std::env::temp_dir(),
            status: Mutex::new(Status::default()),
            engine: Mutex::new(None),
        })
    }

    async fn json_body(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/ping").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "pong");
    }

    #[tokio::test]
    async fn status_is_starting_before_load() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["state"], "starting");
    }

    #[tokio::test]
    async fn cancel_acks() {
        let resp = router(test_state())
            .oneshot(Request::post("/v1/cancel").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "Cancelled");
    }

    #[tokio::test]
    async fn transcribe_before_ready_conflicts() {
        let body =
            serde_json::to_vec(&json!({ "audio_data": [0.0f32, 0.1], "sample_rate": 16000 }))
                .unwrap();
        let resp = router(test_state())
            .oneshot(
                Request::post("/v1/transcribe")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "not_ready");
    }

    #[test]
    fn load_state_wire_strings() {
        assert_eq!(LoadState::Starting.as_str(), "starting");
        assert_eq!(LoadState::Loading.as_str(), "loading");
        assert_eq!(LoadState::Ready.as_str(), "ready");
        assert_eq!(LoadState::Error.as_str(), "error");
    }

    #[tokio::test]
    async fn status_includes_populated_fields() {
        let state = test_state();
        {
            let mut st = state.status.lock().unwrap();
            st.state = LoadState::Ready;
            st.model = Some("voxtral-mini".to_string());
            st.device = Some("cuda".to_string());
            st.reason = Some("recovered".to_string());
        }
        let resp = router(state)
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let v = json_body(resp).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["model"]["name"], "voxtral-mini");
        assert_eq!(v["device"], "cuda");
        assert_eq!(v["reason"], "recovered");
    }

    #[tokio::test]
    async fn load_sets_model_name_synchronously() {
        // The handler sets the model name before spawning the load task, and the
        // error path leaves it intact — so it's readable right after the 202.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini", "device": "cpu" })).unwrap();
        let resp = router(Arc::clone(&state))
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            state.status.lock().unwrap().model.as_deref(),
            Some("voxtral-mini")
        );
    }

    #[tokio::test]
    async fn transcribe_empty_audio_is_invalid() {
        // Ready state, but empty audio_data → 400 invalid_audio (contract), before
        // the engine is ever touched (so no model needed to exercise it).
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Ready;
        let body = serde_json::to_vec(&json!({ "audio_data": [], "sample_rate": 16000 })).unwrap();
        let resp = router(state)
            .oneshot(
                Request::post("/v1/transcribe")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["message"], "invalid_audio");
    }

    #[tokio::test]
    async fn load_missing_weights_transitions_to_error() {
        // backend_dir is a temp dir with no models/, so the load fails fast.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini", "device": "cpu" })).unwrap();
        let resp = router(Arc::clone(&state))
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        // The load runs in a spawned task; with no weights it must reach `error`.
        for _ in 0..250 {
            if matches!(state.status.lock().unwrap().state, LoadState::Error) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("load did not reach error state");
    }
}
