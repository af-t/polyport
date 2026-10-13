use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::json;

pub async fn embeddings_handler() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "Embeddings are not supported by this proxy: no embedding-capable upstream is configured.",
                "type": "not_implemented_error",
                "code": "embeddings_unsupported"
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::app::{AppState, create_app};

    #[tokio::test]
    async fn test_embeddings_handler_returns_501_body() {
        let state = AppState::default();
        let app = create_app(state);
        let req = Request::builder()
            .uri("/v1/embeddings")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(res.into_body(), 1024 * 16).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "embeddings_unsupported");
    }
}
