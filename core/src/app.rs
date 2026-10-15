use std::sync::Arc;
use tokio::sync::RwLock;

use axum::Router;
use axum::routing::{get, post};

use crate::routes::{embeddings, health, models, oauth, proxy};
use crate::services::account_pool::AccountPool;

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<RwLock<AccountPool>>,
    pub http_client: reqwest::Client,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            pool: Arc::new(RwLock::new(AccountPool::default())),
            http_client: reqwest::Client::new(),
        }
    }
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health_handler))
        .route("/healt", get(health::health_handler))
        .route("/v1/models", get(models::models_handler))
        .route("/api/v1/models", get(models::models_handler))
        .route("/v1/models/{id}", get(models::model_retrieve_handler))
        .route("/api/v1/models/{id}", get(models::model_retrieve_handler))
        .route("/v1/embeddings", post(embeddings::embeddings_handler))
        .route("/api/v1/embeddings", post(embeddings::embeddings_handler))
        .route("/oauth2callback", get(oauth::oauth_callback_handler))
        .route("/callback", get(oauth::claude_callback_handler))
        .route("/auth/callback", get(oauth::codex_callback_handler))
        .route("/v1/messages", post(proxy::v1_messages_handler))
        .route("/api/v1/messages", post(proxy::v1_messages_handler))
        .route("/v1/chat/completions", post(proxy::v1_chat_completions_handler))
        .route("/api/v1/chat/completions", post(proxy::v1_chat_completions_handler))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_route_returns_ok_200() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/health")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_routes_return_catalog_json() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_list_returns_openai_list_shape() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/api/v1/models")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val.get("object").and_then(serde_json::Value::as_str), Some("list"));
        let data = val.get("data").and_then(serde_json::Value::as_array).unwrap();
        assert_ne!(data.len(), 0);
        let first = &data[0];
        assert_eq!(first.get("object").and_then(serde_json::Value::as_str), Some("model"));
        assert!(first.get("id").and_then(serde_json::Value::as_str).is_some());
        assert!(first.get("owned_by").and_then(serde_json::Value::as_str).is_some());
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/nonexistent-endpoint")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_embeddings_returns_501_not_implemented() {
        let state = AppState::default();
        let app = create_app(state);
        for uri in ["/v1/embeddings", "/api/v1/embeddings"] {
            let req = Request::builder()
                .uri(uri)
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from("{}"))
                .unwrap();
            let res = app.clone().oneshot(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED, "Failed for {uri}");
        }
    }

    #[tokio::test]
    async fn test_healt_alias_returns_ok() {
        let state = AppState::default();
        let app = create_app(state);
        let req = Request::builder()
            .uri("/healt")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_model_retrieve_returns_object_or_404() {
        let state = AppState::default();
        let app = create_app(state);
        let req = Request::builder()
            .uri("/v1/models/gemini-3.5-flash")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let found = app.clone().oneshot(req).await.unwrap();
        assert_eq!(found.status(), StatusCode::OK);
        let unknown = Request::builder()
            .uri("/v1/models/does-not-exist-xyz")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let absent = app.oneshot(unknown).await.unwrap();
        assert_eq!(absent.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_callback_routes_and_api_v1_aliases_registered() {
        let state = AppState::default();
        let app = create_app(state);

        let endpoints = [
            ("/callback", "GET"),
            ("/auth/callback", "GET"),
            ("/api/v1/messages", "POST"),
            ("/api/v1/chat/completions", "POST"),
        ];

        for (uri, method) in endpoints {
            let body = if method == "POST" {
                Body::from("{}")
            } else {
                Body::empty()
            };
            let mut req = Request::builder().uri(uri).method(method);
            if method == "POST" {
                req = req.header("Content-Type", "application/json");
            }
            let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
            assert_ne!(resp.status(), StatusCode::NOT_FOUND, "Failed for {uri}");
        }
    }
}
