use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::Value;

use crate::app::AppState;
use crate::model_mapping::{MappingEndpoint, resolve_route_candidates};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub system: Option<Value>,
    pub stream: Option<bool>,
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionsRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub stream: Option<bool>,
    pub effort: Option<String>,
}

pub async fn v1_messages_handler(
    State(_state): State<AppState>,
    body: Result<Json<MessagesRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Ok(Json(req)) = body else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Invalid JSON request body", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let Some(model) = req.model else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Missing required field: model", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let _candidates = match resolve_route_candidates(&model, MappingEndpoint::Messages, req.effort.as_deref()) {
        Ok(c) => c,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": err.to_string(), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{ "type": "text", "text": "OK" }]
        })),
    )
        .into_response()
}

pub async fn v1_chat_completions_handler(
    State(_state): State<AppState>,
    body: Result<Json<CompletionsRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Ok(Json(req)) = body else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Invalid JSON request body", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let Some(model) = req.model else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Missing required field: model", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let _candidates = match resolve_route_candidates(&model, MappingEndpoint::OpenAi, req.effort.as_deref()) {
        Ok(c) => c,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": err.to_string(), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "OK" },
                "finish_reason": "stop"
            }]
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::app::{AppState, create_app};

    #[tokio::test]
    async fn test_messages_endpoint_rejects_missing_model() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_completions_endpoint_rejects_invalid_body() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from("not-json"))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_proxy_routes_route_candidates_resolution() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "unknown-nonexistent-model",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
