use axum::Router;
use axum::handler::Handler;
use axum::routing::get;
use polyport_core::app::{AppState, create_app};

use crate::file_api::{FileState, file_handler};
use crate::static_files::{UiState, static_fallback};

pub fn build_webui_app(core_state: AppState, ui_state: UiState, file_state: FileState, proxy_only: bool) -> Router {
    let mut app = create_app(core_state).route("/api/file", get(file_handler).with_state(file_state));
    if !proxy_only {
        app = app.fallback_service(static_fallback.with_state(ui_state));
    }
    app
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use polyport_core::app::AppState;
    use tower::ServiceExt;

    use super::build_webui_app;
    use crate::file_api::FileState;
    use crate::static_files::UiState;

    #[tokio::test]
    async fn test_proxy_only_serves_no_ui_bytes() {
        unsafe {
            std::env::set_var("POLYPORT_PROXY_ONLY", "1");
        }
        let temp_dir = tempfile::tempdir().unwrap();
        let core_state = AppState::default();
        let ui_state = UiState {
            asset_dir: Some(temp_dir.path().to_path_buf()),
        };
        let file_state = FileState {
            sandbox_root: temp_dir.path().to_path_buf(),
        };
        let app = build_webui_app(core_state, ui_state, file_state, true);
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        unsafe {
            std::env::remove_var("POLYPORT_PROXY_ONLY");
        }
    }

    #[tokio::test]
    async fn test_ui_served_when_not_proxy_only() {
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::write(temp_dir.path().join("index.html"), "<h1>polyport ui</h1>").unwrap();
        let core_state = AppState::default();
        let ui_state = UiState {
            asset_dir: Some(temp_dir.path().to_path_buf()),
        };
        let file_state = FileState {
            sandbox_root: temp_dir.path().to_path_buf(),
        };
        let app = build_webui_app(core_state, ui_state, file_state, false);
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        assert!(body.windows(15).any(|w| w == b"<h1>polyport ui"));
    }

    #[tokio::test]
    async fn test_file_api_mounted_on_app() {
        let temp_dir = tempfile::tempdir().unwrap();
        let core_state = AppState::default();
        let ui_state = UiState {
            asset_dir: Some(temp_dir.path().to_path_buf()),
        };
        let file_state = FileState {
            sandbox_root: temp_dir.path().to_path_buf(),
        };
        let app = build_webui_app(core_state, ui_state, file_state, false);
        let req = Request::builder().uri("/api/file?path=").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_core_health_mounted_on_app() {
        let temp_dir = tempfile::tempdir().unwrap();
        let core_state = AppState::default();
        let ui_state = UiState {
            asset_dir: Some(temp_dir.path().to_path_buf()),
        };
        let file_state = FileState {
            sandbox_root: temp_dir.path().to_path_buf(),
        };
        let app = build_webui_app(core_state, ui_state, file_state, false);
        let req = Request::builder().uri("/health").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
