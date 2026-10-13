use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::models::models_catalog_json;

pub async fn models_handler() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(models_catalog_json()))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}
