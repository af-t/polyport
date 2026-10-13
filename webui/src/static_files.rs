use std::path::{Path, PathBuf};

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "public/"]
struct EmbeddedAssets;

#[derive(Clone, Debug)]
pub struct UiState {
    pub asset_dir: Option<PathBuf>,
}

#[allow(clippy::unused_async)]
pub async fn static_fallback(State(state): State<UiState>, req: Request<Body>) -> Response {
    let raw = req.uri().path();
    let path = if raw.is_empty() || raw == "/" {
        "/index.html"
    } else {
        raw
    };
    if path.split('/').any(|segment| segment == "..") {
        return not_found(&req);
    }
    let relative = path.trim_start_matches('/').to_owned();
    if let Some(dir) = std::env::var_os("POLYPORT_PUBLIC_DIR").map(PathBuf::from)
        && let Some(res) = serve_from_disk(&dir, &relative)
    {
        return res;
    }
    if let Some(dir) = state.asset_dir.as_ref()
        && let Some(res) = serve_from_disk(dir, &relative)
    {
        return res;
    }
    if let Some(file) = EmbeddedAssets::get(&relative) {
        return ok_response(file.data.into_owned(), &relative);
    }
    not_found(&req)
}

fn serve_from_disk(dir: &Path, relative: &str) -> Option<Response> {
    let bytes = std::fs::read(dir.join(relative)).ok()?;
    Some(ok_response(bytes, relative))
}

fn ok_response(bytes: Vec<u8>, path: &str) -> Response {
    let mime = mime_guess::from_path(path).first_or_octet_stream().to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", mime)
        .body(Body::from(bytes))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn not_found(req: &Request<Body>) -> Response {
    let wants_html = req
        .headers()
        .get("accept")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/html"));
    if wants_html {
        let page = "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">\
            <title>404 - Page Not Found | Polyport</title>\
            <style>body{background:#0f0f10;color:#e3e3e4;font-family:sans-serif;text-align:center;padding:3rem}</style>\
            </head><body><h1>404 - Page Not Found | Polyport</h1>\
            <p>The page you are looking for could not be found.</p></body></html>";
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "text/html; charset=utf-8")
            .body(Body::from(page))
            .unwrap_or_else(|_| StatusCode::NOT_FOUND.into_response())
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::{UiState, static_fallback};

    #[tokio::test]
    async fn test_serves_index_at_root_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.html"), "<h1>hi</h1>").unwrap();
        let state = UiState {
            asset_dir: Some(dir.path().to_path_buf()),
        };
        let req = axum::http::Request::builder()
            .uri("/")
            .body(axum::body::Body::empty())
            .unwrap();
        let res = static_fallback(axum::extract::State(state), req).await;
        assert_eq!(res.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        assert!(body.windows(11).any(|w| w == b"<h1>hi</h1>"));
    }

    #[tokio::test]
    async fn test_styled_404_for_html_accept() {
        let state = UiState { asset_dir: None };
        let req = axum::http::Request::builder()
            .uri("/nope")
            .header("accept", "text/html")
            .body(axum::body::Body::empty())
            .unwrap();
        let res = static_fallback(axum::extract::State(state), req).await;
        assert_eq!(res.status(), axum::http::StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("404 - Page Not Found | Polyport"));
    }

    #[tokio::test]
    async fn test_plain_404_otherwise() {
        let state = UiState { asset_dir: None };
        let req = axum::http::Request::builder()
            .uri("/nope")
            .body(axum::body::Body::empty())
            .unwrap();
        let res = static_fallback(axum::extract::State(state), req).await;
        assert_eq!(res.status(), axum::http::StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        assert!(body.is_empty());
    }
}
