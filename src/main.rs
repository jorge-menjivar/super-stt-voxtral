// SPDX-License-Identifier: GPL-3.0-only
//! Voxtral subprocess backend: serves the Super STT `/v1` contract over a
//! pathname Unix socket (`SUPER_STT_BACKEND_SOCKET`), loading the model from
//! `SUPER_STT_BACKEND_DIR/models/<name>`. Self-contained — no super-stt deps.
//!
//! The model runs on Burn, whose GPU kernels CubeCL compiles at runtime and
//! keeps in `SUPER_STT_BACKEND_CACHE_DIR` — the one writable directory a
//! daemon new enough to grant it provides.

// doc lint trips on prose like "candle"/"super-stt".
#![allow(clippy::doc_markdown)]

mod inference;
mod progress;
mod voxtral;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

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
use progress::Report;

/// The models `backend.toml` declares, the only names a load or a prepare
/// accepts.
const MODELS: [&str; 2] = ["voxtral-mini-3b-2507", "voxtral-small-24b-2507"];

#[derive(Clone, Copy)]
enum LoadState {
    Starting,
    Preparing,
    Prepared,
    Loading,
    Ready,
    Error,
}

impl LoadState {
    fn as_str(self) -> &'static str {
        match self {
            LoadState::Starting => "starting",
            LoadState::Preparing => "preparing",
            LoadState::Prepared => "prepared",
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
    /// How far a prepare or a load has got; reported only while `state` is
    /// `preparing` or `loading`.
    load: Report,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: LoadState::Starting,
            model: None,
            device: None,
            reason: None,
            load: Report::default(),
        }
    }
}

struct AppState {
    backend_dir: PathBuf,
    status: Mutex<Status>,
    engine: Mutex<Option<VoxtralEngine>>,
    /// The prepare in flight, which a load waits for before it maps any
    /// weights.
    prepare: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// Names the one writable directory the sandbox grants, where the compiled
/// kernels go. Absent when the daemon grants none.
const ENV_CACHE_DIR: &str = "SUPER_STT_BACKEND_CACHE_DIR";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // CubeCL's ROCm compiler logs its whole IR after every pass at `info`,
    // and the daemon runs backends at `info`. So it starts at `warn`, which
    // `RUST_LOG` can still raise by naming it.
    env_logger::Builder::new()
        .filter_module("pliron", log::LevelFilter::Warn)
        .parse_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // Before anything touches a device: CubeCL's configuration, which says
    // where compiled kernels are kept and which stream work runs on, is frozen
    // the first time it is read.
    let cache_dir = std::env::var_os(ENV_CACHE_DIR).map(PathBuf::from);
    match &cache_dir {
        Some(dir) => log::info!("keeping compiled kernels in {}", dir.display()),
        None => log::warn!(
            "{ENV_CACHE_DIR} is not set, so the GPU kernels have nowhere to be kept and are \
             recompiled on every load"
        ),
    }
    inference::configure_cubecl(cache_dir.as_deref());

    let socket = std::env::var("SUPER_STT_BACKEND_SOCKET")
        .context("SUPER_STT_BACKEND_SOCKET must be set")?;
    let backend_dir =
        std::env::var("SUPER_STT_BACKEND_DIR").context("SUPER_STT_BACKEND_DIR must be set")?;

    let state = Arc::new(AppState {
        backend_dir: PathBuf::from(backend_dir),
        status: Mutex::new(Status::default()),
        engine: Mutex::new(None),
        prepare: Mutex::new(None),
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
        .route("/v1/prepare", post(prepare))
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
    if matches!(st.state, LoadState::Preparing | LoadState::Loading) {
        if let Some(phase) = st.load.phase {
            out["phase"] = json!(phase.as_str());
        }
        if let Some(step) = st.load.step {
            out["step"] = json!(step.as_str());
        }
        if let Some(progress) = st.load.progress {
            out["progress"] = json!(progress);
        }
    }
    Json(out)
}

#[derive(Deserialize)]
struct LoadReq {
    name: String,
    #[serde(default)]
    device: Option<String>,
}

/// Build the kernels a load of `name` will need from the model's small files,
/// while its weights are still downloading.
///
/// It never fails a load: a prepare that errors or panics is logged and goes
/// back to `starting`, and the load then builds the kernels itself.
async fn prepare(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    // Contract: a model this backend does not declare is `invalid_model`.
    if !MODELS.contains(&req.name.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_model" })),
        );
    }
    let accepted = (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Preparing started" })),
    );
    let dir = s.backend_dir.join("models").join(&req.name);
    {
        let mut st = s.status.lock().unwrap();
        match st.state {
            LoadState::Loading | LoadState::Ready => {
                return (
                    StatusCode::CONFLICT,
                    Json(json!({ "status": "error", "message": "already_loading" })),
                );
            }
            LoadState::Preparing | LoadState::Prepared => return accepted,
            LoadState::Starting | LoadState::Error => {}
        }
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
        st.load = Report::default();
        if VoxtralEngine::kernels_built(&dir) {
            st.state = LoadState::Prepared;
            return accepted;
        }
        st.state = LoadState::Preparing;
    }
    let device = req.device;
    let s2 = Arc::clone(&s);
    let handle = tokio::spawn(async move {
        let s3 = Arc::clone(&s2);
        let built = tokio::task::spawn_blocking(move || {
            let report = |load: Report| {
                let mut st = s3.status.lock().unwrap_or_else(PoisonError::into_inner);
                if matches!(st.state, LoadState::Preparing | LoadState::Loading) {
                    st.load = load;
                }
            };
            VoxtralEngine::build_kernels(&dir, device.as_deref(), &report)
        })
        .await;
        let built = match built {
            Ok(Ok(())) => {
                log::info!("kernels built; prepared");
                true
            }
            Ok(Err(e)) => {
                log::warn!("building the kernels ahead of the load failed: {e:#}");
                false
            }
            Err(e) => {
                log::warn!("building the kernels ahead of the load panicked: {e}");
                false
            }
        };
        let mut st = s2.status.lock().unwrap_or_else(PoisonError::into_inner);
        // A load that arrived meanwhile owns the status now.
        if matches!(st.state, LoadState::Preparing) {
            st.state = if built {
                LoadState::Prepared
            } else {
                LoadState::Starting
            };
            st.load = Report::default();
        }
    });
    *s.prepare.lock().unwrap() = Some(handle);
    accepted
}

async fn load(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    // Contract: a model this backend does not declare is `invalid_model`.
    if !MODELS.contains(&req.name.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_model" })),
        );
    }
    let preparing = {
        let mut st = s.status.lock().unwrap();
        // Contract: reject a concurrent load. A model switch is a fresh load
        // after the daemon tears this backend down, so only an in-flight load on
        // this instance trips `already_loading`.
        if matches!(st.state, LoadState::Loading) {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "status": "error", "message": "already_loading" })),
            );
        }
        // A prepare still building goes on reporting its progress, as this
        // load's, and the whole load then counts as the initial setup it is.
        let preparing = matches!(st.state, LoadState::Preparing);
        st.state = LoadState::Loading;
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
        if !preparing {
            st.load = Report::default();
        }
        preparing
    };
    let dir = s.backend_dir.join("models").join(&req.name);
    let device = req.device;
    let pending = s.prepare.lock().unwrap().take();
    let s2 = Arc::clone(&s);
    tokio::spawn(async move {
        // The prepare hands its memory back before the weights are mapped.
        // Its outcome does not matter: the load builds whatever it did not.
        if let Some(pending) = pending {
            let _ = pending.await;
        }
        let s3 = Arc::clone(&s2);
        // Everything that can panic runs in the blocking task, whose panic
        // comes back as an error below. A panic in this task itself would end
        // it with the status stuck at `loading`.
        let res = tokio::task::spawn_blocking(move || {
            // A transcription that panicked while holding the engine poisoned
            // the lock; a load replaces that engine all the same, and clears
            // the poison once it has.
            let mut engine = s3.engine.lock().unwrap_or_else(PoisonError::into_inner);
            // The previous model goes first, so its memory is back before the
            // new one asks for any.
            *engine = None;
            s3.engine.clear_poison();
            let report = |mut load: Report| {
                if preparing {
                    load.phase = Some(progress::Phase::InitialSetup);
                }
                let mut st = s3.status.lock().unwrap_or_else(PoisonError::into_inner);
                if matches!(st.state, LoadState::Loading) {
                    st.load = load;
                }
            };
            let loaded = VoxtralEngine::load(&dir, device.as_deref(), &report)?;
            let label = loaded.device_label().to_string();
            *engine = Some(loaded);
            anyhow::Ok(label)
        })
        .await;
        match res {
            Ok(Ok(label)) => {
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
    #[serde(default)]
    language: Option<String>,
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
    let language = req.language;
    let s2 = Arc::clone(&s);
    // The reply goes out before the engine hands its working memory back, which
    // the caller has no reason to wait for. The lock is held until that is done,
    // so a request arriving meanwhile waits for it rather than racing it.
    let (reply, replied) = tokio::sync::oneshot::channel();
    let worker = tokio::task::spawn_blocking(move || {
        let mut guard = s2.engine.lock().unwrap();
        let Some(engine) = guard.as_mut() else {
            let _ = reply.send(Err(anyhow::anyhow!("engine not loaded")));
            return;
        };
        // Released whether or not the transcription failed: a failed request's
        // memory is just as dead.
        let _ = reply.send(engine.transcribe(&audio, sample_rate, language.as_deref()));
        engine.release_memory();
    });
    let result = match replied.await {
        Ok(result) => Ok(result),
        // The reply was dropped unsent, which only a panic does: the join
        // error carries it.
        Err(_) => worker
            .await
            .map(|()| Err(anyhow::anyhow!("the transcription ended without a reply"))),
    };
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
        // A task panic is still an inference failure; the contract documents
        // `inference_failed` for 500, so report that (the panic is in `detail`).
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "inference_failed", "detail": format!("panicked: {e}") }),
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
            prepare: Mutex::new(None),
        })
    }

    fn post(path: &str, body: &Value) -> Request<Body> {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    async fn wait_for(state: &AppState, done: impl Fn(LoadState) -> bool) -> LoadState {
        for _ in 0..250 {
            let now = state.status.lock().unwrap().state;
            if done(now) {
                return now;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the state never got there");
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
        assert_eq!(LoadState::Preparing.as_str(), "preparing");
        assert_eq!(LoadState::Prepared.as_str(), "prepared");
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
            st.model = Some("voxtral-mini-3b-2507".to_string());
            st.device = Some("cuda".to_string());
            st.reason = Some("recovered".to_string());
        }
        let resp = router(state)
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let v = json_body(resp).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["model"]["name"], "voxtral-mini-3b-2507");
        assert_eq!(v["device"], "cuda");
        assert_eq!(v["reason"], "recovered");
    }

    #[tokio::test]
    async fn status_reports_load_progress_only_while_loading() {
        let state = test_state();
        {
            let mut st = state.status.lock().unwrap();
            st.state = LoadState::Loading;
            st.load = Report {
                phase: Some(progress::Phase::InitialSetup),
                step: Some(progress::Step::BuildingKernels),
                progress: Some(0.5),
            };
        }
        let get = || Request::get("/v1/status").body(Body::empty()).unwrap();
        let v = json_body(router(Arc::clone(&state)).oneshot(get()).await.unwrap()).await;
        assert_eq!(v["state"], "loading");
        assert_eq!(v["phase"], "initial_setup");
        assert_eq!(v["step"], "building_kernels");
        assert_eq!(v["progress"], 0.5);

        state.status.lock().unwrap().state = LoadState::Preparing;
        let v = json_body(router(Arc::clone(&state)).oneshot(get()).await.unwrap()).await;
        assert_eq!(v["state"], "preparing");
        assert_eq!(v["step"], "building_kernels");
        assert_eq!(v["progress"], 0.5);

        for done in [LoadState::Prepared, LoadState::Ready] {
            state.status.lock().unwrap().state = done;
            let v = json_body(router(Arc::clone(&state)).oneshot(get()).await.unwrap()).await;
            for field in ["phase", "step", "progress"] {
                assert!(v.get(field).is_none(), "{field} outside a load: {v}");
            }
        }
    }

    #[tokio::test]
    async fn prepare_is_refused_once_a_load_has_begun() {
        for begun in [LoadState::Loading, LoadState::Ready] {
            let state = test_state();
            state.status.lock().unwrap().state = begun;
            let resp = router(state)
                .oneshot(post(
                    "/v1/prepare",
                    &json!({ "name": "voxtral-mini-3b-2507" }),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::CONFLICT);
            assert_eq!(json_body(resp).await["message"], "already_loading");
        }
    }

    #[tokio::test]
    async fn prepare_rejects_an_undeclared_model() {
        for name in ["whisper-tiny", "../voxtral-mini-3b-2507"] {
            let resp = router(test_state())
                .oneshot(post("/v1/prepare", &json!({ "name": name })))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            assert_eq!(json_body(resp).await["message"], "invalid_model");
        }
    }

    #[tokio::test]
    async fn a_failed_prepare_goes_back_to_starting() {
        // backend_dir has no models/, so the prepare fails before the device.
        let state = test_state();
        let req = json!({ "name": "voxtral-mini-3b-2507", "device": "cuda" });
        let resp = router(Arc::clone(&state))
            .oneshot(post("/v1/prepare", &req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(json_body(resp).await["message"], "Preparing started");
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Preparing)).await;
        assert!(matches!(ended, LoadState::Starting));
    }

    #[tokio::test]
    async fn a_load_during_a_prepare_waits_for_it_and_keeps_the_status() {
        let state = test_state();
        let req = json!({ "name": "voxtral-mini-3b-2507", "device": "cuda" });
        let app = router(Arc::clone(&state));
        let resp = app
            .clone()
            .oneshot(post("/v1/prepare", &req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let resp = app.oneshot(post("/v1/load", &req)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        // The prepare fails, which the load does not mind; the load then fails
        // on the missing weights, and the prepare's ending does not overwrite
        // that.
        let ended = wait_for(&state, |now| !matches!(now, LoadState::Loading)).await;
        assert!(matches!(ended, LoadState::Error));
        assert!(state.prepare.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn load_rejects_an_undeclared_model() {
        for name in ["whisper-tiny", "../voxtral-mini-3b-2507"] {
            let state = test_state();
            let resp = router(Arc::clone(&state))
                .oneshot(post("/v1/load", &json!({ "name": name })))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            assert_eq!(json_body(resp).await["message"], "invalid_model");
            assert!(matches!(
                state.status.lock().unwrap().state,
                LoadState::Starting
            ));
        }
    }

    #[tokio::test]
    async fn load_rejects_concurrent_load() {
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Loading;
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini-3b-2507" })).unwrap();
        let resp = router(state)
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "already_loading");
    }

    #[tokio::test]
    async fn load_sets_model_name_synchronously() {
        // The handler sets the model name before spawning the load task, and the
        // error path leaves it intact — so it's readable right after the 202.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini-3b-2507", "device": "cuda" }))
            .unwrap();
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
            Some("voxtral-mini-3b-2507")
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
    async fn load_after_a_poisoned_engine_still_finishes() {
        // A transcription that panicked under the engine lock poisons it. The
        // next load must still reach a final state, and leave the lock usable.
        let state = test_state();
        let s2 = Arc::clone(&state);
        let _ = std::thread::spawn(move || {
            let _guard = s2.engine.lock().unwrap();
            panic!("a transcription panicking under the lock");
        })
        .join();
        assert!(state.engine.is_poisoned());
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini-3b-2507" })).unwrap();
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
        for _ in 0..250 {
            if matches!(state.status.lock().unwrap().state, LoadState::Error) {
                assert!(!state.engine.is_poisoned());
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("load did not reach error state");
    }

    #[tokio::test]
    async fn load_missing_weights_transitions_to_error() {
        // backend_dir is a temp dir with no models/, so the load fails fast.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "voxtral-mini-3b-2507", "device": "cuda" }))
            .unwrap();
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
