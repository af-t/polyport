use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::AppState;
use crate::mappers::gemini::map_messages_to_gemini;
use crate::model_mapping::{MappingEndpoint, resolve_route_candidates};
use crate::services::account_pool::AttemptError;
use crate::services::gemini::{CallGeminiParams, GeminiStreamResult, build_generation_config, call_gemini_stream};
use crate::types::Provider;
use crate::utils::sse::{
    AnthropicStreamState, OpenAiStreamState, collect_gemini_response, gemini_to_anthropic_sse, gemini_to_openai_sse,
    parse_gemini_sse_line,
};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub system: Option<Value>,
    pub stream: Option<bool>,
    pub thinking: Option<Value>,
    pub effort: Option<String>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionsRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub stream: Option<bool>,
    pub effort: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Value>,
    pub tools: Option<Vec<Value>>,
}

fn attempt_error_to_response(err: &AttemptError) -> Response {
    let status = match err {
        AttemptError::RateLimit(rl) => StatusCode::from_u16(rl.status).unwrap_or(StatusCode::TOO_MANY_REQUESTS),
        AttemptError::Upstream(u) => StatusCode::from_u16(u.status).unwrap_or(StatusCode::BAD_GATEWAY),
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(serde_json::json!({
            "error": { "message": err.to_string(), "type": "api_error" }
        })),
    )
        .into_response()
}

fn handle_gemini_messages_stream(stream_res: GeminiStreamResult, model: &str) -> Response {
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(32);
    let model_name = model.to_string();
    tokio::spawn(async move {
        let mut stream = stream_res.response.bytes_stream();
        let mut buffer = String::new();
        let mut sse_state = AnthropicStreamState::new(&format!("msg_{}", uuid::Uuid::new_v4()), &model_name);
        while let Some(Ok(bytes)) = stream.next().await {
            if let Ok(chunk_str) = std::str::from_utf8(&bytes) {
                buffer.push_str(chunk_str);
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer.drain(..=pos);
                    if let Some(parsed) = parse_gemini_sse_line(&line) {
                        let events = gemini_to_anthropic_sse(&parsed, &mut sse_state);
                        for ev in events {
                            let _ = tx.send(Ok(axum::body::Bytes::from(ev))).await;
                        }
                    }
                }
            }
        }
    });
    let body = axum::body::Body::from_stream(rx);
    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}

async fn handle_gemini_messages_buffered(stream_res: GeminiStreamResult, model: &str) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_gemini_sse_line).collect();
    let (full_text, finish_reason, prompt_tokens, output_tokens) = collect_gemini_response(&chunks);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{ "type": "text", "text": full_text }],
            "stop_reason": finish_reason.unwrap_or_else(|| "end_turn".to_string()),
            "stop_sequence": serde_json::Value::Null,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": output_tokens
            }
        })),
    )
        .into_response()
}

pub async fn v1_messages_handler(
    State(state): State<AppState>,
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

    let Some(model) = req.model.clone() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Missing required field: model", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let candidates = match resolve_route_candidates(&model, MappingEndpoint::Messages, req.effort.as_deref()) {
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

    let Some(candidate) = candidates.first() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "No route candidate available", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    if candidate.route.provider != Provider::Gemini {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": { "message": "Provider not yet connected", "type": "not_implemented" }
            })),
        )
            .into_response();
    }

    let is_thinking_requested = req.thinking.is_some() || req.effort.is_some();
    let system_str = match &req.system {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let raw_val = serde_json::to_value(&req).unwrap_or_default();
    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();

    let (system_instruction, contents) = match map_messages_to_gemini(&messages_val, system_str, is_thinking_requested)
    {
        Ok(res) => res,
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

    let generation_config = build_generation_config(&raw_val, &candidate.route.upstream_model, is_thinking_requested);

    let params = CallGeminiParams {
        contents: contents
            .into_iter()
            .map(|c| serde_json::to_value(c).unwrap_or_default())
            .collect(),
        system_instruction: system_instruction.map(|s| serde_json::to_value(s).unwrap_or_default()),
        generation_config,
        tools: req.tools.clone(),
        tool_config: req.tool_choice.clone(),
        upstream_model: candidate.route.upstream_model.clone(),
        model_class: candidate.route.model_class,
        preferred_account_id: None,
    };

    match call_gemini_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            if req.stream.unwrap_or(false) {
                handle_gemini_messages_stream(stream_res, &model)
            } else {
                handle_gemini_messages_buffered(stream_res, &model).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

fn handle_gemini_completions_stream(stream_res: GeminiStreamResult, model: &str) -> Response {
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(32);
    let model_name = model.to_string();
    tokio::spawn(async move {
        let mut stream = stream_res.response.bytes_stream();
        let mut buffer = String::new();
        let mut sse_state = OpenAiStreamState::new(&format!("chatcmpl-{}", uuid::Uuid::new_v4()), &model_name);
        while let Some(Ok(bytes)) = stream.next().await {
            if let Ok(chunk_str) = std::str::from_utf8(&bytes) {
                buffer.push_str(chunk_str);
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer.drain(..=pos);
                    if let Some(parsed) = parse_gemini_sse_line(&line) {
                        let events = gemini_to_openai_sse(&parsed, &mut sse_state);
                        for ev in events {
                            let _ = tx.send(Ok(axum::body::Bytes::from(ev))).await;
                        }
                    }
                }
            }
        }
    });
    let body = axum::body::Body::from_stream(rx);
    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}

async fn handle_gemini_completions_buffered(stream_res: GeminiStreamResult, model: &str) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_gemini_sse_line).collect();
    let (full_text, _finish_reason, prompt_tokens, output_tokens) = collect_gemini_response(&chunks);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": full_text },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens
            }
        })),
    )
        .into_response()
}

pub async fn v1_chat_completions_handler(
    State(state): State<AppState>,
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

    let Some(model) = req.model.clone() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "Missing required field: model", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    let candidates = match resolve_route_candidates(&model, MappingEndpoint::OpenAi, req.effort.as_deref()) {
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

    let Some(candidate) = candidates.first() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "message": "No route candidate available", "type": "invalid_request_error" }
            })),
        )
            .into_response();
    };

    if candidate.route.provider != Provider::Gemini {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": { "message": "Provider not yet connected", "type": "not_implemented" }
            })),
        )
            .into_response();
    }

    let is_thinking_requested = req.effort.is_some();
    let raw_val = serde_json::to_value(&req).unwrap_or_default();
    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();

    let (system_instruction, contents) = match map_messages_to_gemini(&messages_val, None, is_thinking_requested) {
        Ok(res) => res,
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

    let generation_config = build_generation_config(&raw_val, &candidate.route.upstream_model, is_thinking_requested);

    let params = CallGeminiParams {
        contents: contents
            .into_iter()
            .map(|c| serde_json::to_value(c).unwrap_or_default())
            .collect(),
        system_instruction: system_instruction.map(|s| serde_json::to_value(s).unwrap_or_default()),
        generation_config,
        tools: req.tools.clone(),
        tool_config: None,
        upstream_model: candidate.route.upstream_model.clone(),
        model_class: candidate.route.model_class,
        preferred_account_id: None,
    };

    match call_gemini_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            if req.stream.unwrap_or(false) {
                handle_gemini_completions_stream(stream_res, &model)
            } else {
                handle_gemini_completions_buffered(stream_res, &model).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
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

    #[tokio::test]
    async fn test_messages_handler_routes_and_dispatches() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "gemini-3.5-flash",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_completions_handler_routes_and_dispatches() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "gemini-3.5-flash",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }
}
