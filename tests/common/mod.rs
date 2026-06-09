// SPDX-License-Identifier: GPL-3.0-only
//! Minimal "mock daemon": spawns the real backend binary and drives the `/v1`
//! contract over the Unix socket the same way the daemon does. Self-contained —
//! no `super-stt` dependency.

#![allow(dead_code)] // each test file uses a subset

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::net::UnixStream;

static SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct Backend {
    child: Child,
    socket: PathBuf,
}

impl Backend {
    /// Spawn the backend binary against `backend_dir`; wait until `/v1/ping` answers.
    pub async fn spawn(backend_dir: &Path) -> Backend {
        let idx = SOCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
        let socket =
            std::env::temp_dir().join(format!("voxtral-test-{}-{idx}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let child = Command::new(env!("CARGO_BIN_EXE_super-stt-backend-voxtral"))
            .env("SUPER_STT_BACKEND_SOCKET", &socket)
            .env("SUPER_STT_BACKEND_DIR", backend_dir)
            .env("RUST_LOG", "warn")
            .spawn()
            .expect("spawn backend binary");
        let backend = Backend { child, socket };
        // Generous headroom: the freshly-built binary may start slowly on a cold
        // CI runner. A timeout here panics with a clear "did not start" message.
        backend.wait_for_ping(Duration::from_secs(20)).await;
        backend
    }

    async fn wait_for_ping(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok((200, _)) = self.request("GET", "/v1/ping", Vec::new()).await {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "backend did not start in {timeout:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// One HTTP request over the Unix socket. Returns `(status, body_bytes)`.
    pub async fn request(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
    ) -> std::io::Result<(u16, Vec<u8>)> {
        let stream = UnixStream::connect(&self.socket).await?;
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(std::io::Error::other)?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        let req = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header("host", "backend.local")
            .header("content-type", "application/json")
            .header("x-stt-model", "voxtral-mini")
            .body(Full::new(Bytes::from(body)))
            .unwrap();
        let resp = sender
            .send_request(req)
            .await
            .map_err(std::io::Error::other)?;
        let status = resp.status().as_u16();
        let bytes = resp
            .into_body()
            .collect()
            .await
            .map_err(std::io::Error::other)?
            .to_bytes()
            .to_vec();
        Ok((status, bytes))
    }

    pub async fn status(&self) -> Value {
        let (_, body) = self.request("GET", "/v1/status", Vec::new()).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    pub async fn load(&self, name: &str, device: &str) -> u16 {
        let body =
            serde_json::to_vec(&serde_json::json!({ "name": name, "device": device })).unwrap();
        self.request("POST", "/v1/load", body).await.unwrap().0
    }

    pub async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> (u16, Value) {
        let body = serde_json::to_vec(
            &serde_json::json!({ "audio_data": audio, "sample_rate": sample_rate }),
        )
        .unwrap();
        let (status, body) = self.request("POST", "/v1/transcribe", body).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    /// Poll `/v1/status` until `state` is one of `wanted`, or panic on timeout.
    pub async fn wait_for_state(&self, wanted: &[&str], timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let s = self.status().await;
            if let Some(state) = s.get("state").and_then(Value::as_str)
                && wanted.contains(&state)
            {
                return s;
            }
            assert!(
                Instant::now() < deadline,
                "state not in {wanted:?} within {timeout:?}; last = {s}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}
