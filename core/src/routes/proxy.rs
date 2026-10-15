use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::AppState;
use crate::config::{ModelNotFoundError, RouteDecision};
use crate::mappers::codex::{map_anthropic_to_responses, map_openai_to_responses};
use crate::mappers::gemini::{map_anthropic_tools_for_gemini, map_messages_to_gemini, map_openai_tools_for_gemini};
use crate::model_mapping::{MappingEndpoint, RouteCandidate, get_route_fingerprint, resolve_route_candidates};
use crate::services::account_pool::AttemptError;
use crate::services::anthropic::{
    CallClaudeParams, ClaudeStreamResult, apply_thinking_constraints, call_claude_stream, map_anthropic_model,
    with_claude_code_identity,
};
use crate::services::codex::{CallCodexParams, CodexStreamResult, call_codex_stream};
use crate::services::gemini::{CallGeminiParams, GeminiStreamResult, build_generation_config, call_gemini_stream};
use crate::types::{ModelClass, Provider};
use crate::utils::completion_cache::{
    completion_key, lookup_completion, replay_completions_json, replay_completions_stream, replay_messages_json,
    replay_messages_stream,
};
use crate::utils::history_cache::{mint_reasoning_token, record_history, record_tool_signature};
use crate::utils::sse::reasoning_format_for;
use crate::utils::sse::{
    AnthropicStreamState, OpenAiStreamState, anthropic_to_openai_sse, codex_to_anthropic_sse, codex_to_openai_sse,
    collect_codex_response, collect_gemini_response, gemini_to_anthropic_sse, gemini_to_openai_sse,
    parse_anthropic_sse_line, parse_codex_sse_line, parse_gemini_sse_line,
};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub system: Option<Value>,
    pub stream: Option<bool>,
    pub thinking: Option<Value>,
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionsRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Value>,
    pub tools: Option<Vec<Value>>,
}

fn reasoning_excluded(reasoning: Option<&Value>) -> bool {
    reasoning
        .and_then(|r| r.get("exclude"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn gemini_history_key(model: Option<&str>, messages: &Value, system: &Value, tools: &Value) -> String {
    crate::utils::history_cache::history_key(model.unwrap_or(""), messages, system, tools)
}

fn requested_effort(
    effort: Option<&str>,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
    output_config: Option<&Value>,
) -> Option<String> {
    if let Some(e) = effort {
        return Some(e.to_string());
    }
    if let Some(e) = reasoning_effort {
        return Some(e.to_string());
    }
    if let Some(e) = reasoning.and_then(|r| r.get("effort")).and_then(Value::as_str) {
        return Some(e.to_string());
    }
    output_config
        .and_then(|o| o.get("effort"))
        .and_then(Value::as_str)
        .map(str::to_string)
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

fn stream_sse_transformed<S, F>(
    mut byte_stream: impl futures::Stream<Item = Result<axum::body::Bytes, reqwest::Error>> + Unpin + Send + 'static,
    mut state: S,
    mut transform: F,
    on_done: Option<Box<dyn FnOnce(S) + Send + 'static>>,
) -> Response
where
    S: Send + 'static,
    F: FnMut(&str, &mut S) -> Vec<String> + Send + 'static,
{
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(32);
    tokio::spawn(async move {
        let mut byte_buf = Vec::new();
        while let Some(Ok(bytes)) = byte_stream.next().await {
            byte_buf.extend_from_slice(&bytes);
            while let Some(pos) = byte_buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = byte_buf.drain(..=pos).collect();
                if let Ok(line) = std::str::from_utf8(&line_bytes) {
                    let events = transform(line, &mut state);
                    for ev in events {
                        let _ = tx.send(Ok(axum::body::Bytes::from(ev))).await;
                    }
                }
            }
        }
        let line_opt = std::str::from_utf8(&byte_buf).ok().filter(|s| !s.is_empty());
        if let Some(line) = line_opt {
            let events = transform(line, &mut state);
            for ev in events {
                let _ = tx.send(Ok(axum::body::Bytes::from(ev))).await;
            }
        }
        if let Some(done) = on_done {
            done(state);
        }
    });
    let body = axum::body::Body::from_stream(rx);
    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}

fn handle_gemini_messages_stream(
    stream_res: GeminiStreamResult,
    model: &str,
    history_key: &str,
    cache_key: &str,
) -> Response {
    let mut sse_state = AnthropicStreamState::new(&format!("msg_{}", uuid::Uuid::new_v4()), model);
    sse_state.history_key = history_key.to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let cache_key = cache_key.to_string();
    stream_sse_transformed(
        stream_res.response.bytes_stream(),
        sse_state,
        move |line, state: &mut AnthropicStreamState| {
            if let Some(parsed) = parse_gemini_sse_line(line) {
                gemini_to_anthropic_sse(&parsed, state)
            } else {
                Vec::new()
            }
        },
        Some(Box::new(move |state: AnthropicStreamState| {
            if let Some(entry) = crate::utils::completion_cache::gemini_stream_entry_from_anthropic(&state, now) {
                crate::utils::completion_cache::write_completion_cache(&cache_key, entry);
            }
        })),
    )
}

async fn handle_gemini_messages_buffered(
    stream_res: GeminiStreamResult,
    model: &str,
    history_key: &str,
    cache_key: &str,
) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_gemini_sse_line).collect();
    let res = collect_gemini_response(&chunks);
    if let Some(sig) = res.thought_signature.as_deref() {
        record_history(history_key, sig, &res.reasoning);
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Some(entry) = crate::utils::completion_cache::gemini_completion_entry(&res, model, now) {
        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
    }

    let mut content = Vec::new();
    if !res.text.is_empty() {
        content.push(serde_json::json!({ "type": "text", "text": res.text }));
    }
    let stop_reason = crate::utils::completion_cache::map_gemini_to_anthropic_stop(
        res.finish_reason.as_deref(),
        !res.function_calls.is_empty(),
    );
    if !res.function_calls.is_empty() {
        for fc in res.function_calls {
            let tool_id = format!("call_{}", uuid::Uuid::new_v4().simple());
            if let Some(sig) = res.thought_signature.as_deref() {
                record_tool_signature(&tool_id, sig, "");
            }
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": tool_id,
                "name": fc.name,
                "input": fc.args,
            }));
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": serde_json::Value::Null,
            "usage": {
                "input_tokens": res.prompt_tokens,
                "output_tokens": res.output_tokens,
            }
        })),
    )
        .into_response()
}

async fn dispatch_gemini_messages(
    state: &AppState,
    req: &MessagesRequest,
    route: &RouteDecision,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    cache_key: &str,
) -> Response {
    let is_thinking_requested = req.thinking.is_some()
        || requested_effort(
            req.effort.as_deref(),
            req.reasoning_effort.as_deref(),
            req.reasoning.as_ref(),
            req.output_config.as_ref(),
        )
        .is_some()
        || crate::config::model_supports_thinking(&route.upstream_model);
    let mut system_buf = String::new();
    if let Some(sys) = &req.system {
        if let Some(s) = sys.as_str() {
            system_buf.push_str(s);
        } else if let Some(arr) = sys.as_array() {
            for block in arr {
                if let Some(txt) = block.get("text").and_then(Value::as_str) {
                    if !system_buf.is_empty() {
                        system_buf.push_str("\n\n");
                    }
                    system_buf.push_str(txt);
                }
            }
        }
    }
    let system_str = if system_buf.is_empty() {
        None
    } else {
        Some(system_buf.as_str())
    };
    let raw_val = serde_json::to_value(req).unwrap_or_default();
    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();

    let history_ctx = crate::utils::history_cache::HistoryContext {
        model: req.model.clone().unwrap_or_default(),
        system: req.system.clone().unwrap_or(Value::Null),
        tools: serde_json::to_value(&req.tools).unwrap_or(Value::Null),
    };
    let (system_instruction, contents) =
        match map_messages_to_gemini(&messages_val, system_str, is_thinking_requested, Some(&history_ctx)) {
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

    let generation_config = build_generation_config(&raw_val, &route.upstream_model, is_thinking_requested);

    let params = CallGeminiParams {
        contents: contents
            .into_iter()
            .map(|c| serde_json::to_value(c).unwrap_or_default())
            .collect(),
        system_instruction: system_instruction.map(|s| serde_json::to_value(s).unwrap_or_default()),
        generation_config,
        tools: req.tools.as_deref().and_then(map_anthropic_tools_for_gemini),
        tool_config: req.tool_choice.clone(),
        upstream_model: route.upstream_model.clone(),
        model_class: route.model_class,
        preferred_account_id,
    };

    let model_name = req.model.as_deref().unwrap_or(&route.upstream_model);
    let history_key = gemini_history_key(
        req.model.as_deref(),
        &messages_val,
        &req.system.clone().unwrap_or(Value::Null),
        &serde_json::to_value(&req.tools).unwrap_or(Value::Null),
    );
    match call_gemini_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Gemini, lookup_keys, Some(&stream_res.account_id));
            }
            if req.stream.unwrap_or(false) {
                handle_gemini_messages_stream(stream_res, model_name, &history_key, cache_key)
            } else {
                handle_gemini_messages_buffered(stream_res, model_name, &history_key, cache_key).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

async fn dispatch_claude_messages(
    state: &AppState,
    req: &MessagesRequest,
    route: &RouteDecision,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    headers: HeaderMap,
    cache_key: &str,
) -> Response {
    let mut claude_body = serde_json::to_value(req).unwrap_or_default();
    let mapped_model = map_anthropic_model(&route.upstream_model);
    claude_body["model"] = Value::String(mapped_model);
    claude_body["system"] = Value::Array(with_claude_code_identity(req.system.as_ref()));
    apply_thinking_constraints(&mut claude_body);

    let params = CallClaudeParams {
        body: claude_body,
        model_class: route.model_class,
        preferred_account_id,
        client_headers: Some(headers),
    };

    match call_claude_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Claude, lookup_keys, Some(&stream_res.account_id));
            }
            if req.stream.unwrap_or(false) {
                let body = axum::body::Body::from_stream(stream_res.response.bytes_stream());
                Response::builder()
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .body(body)
                    .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
            } else {
                let bytes = stream_res.response.bytes().await.unwrap_or_default();
                if let Ok(json_val) = serde_json::from_slice::<Value>(&bytes) {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs());
                    let model_name = req.model.clone().unwrap_or_else(|| route.upstream_model.clone());
                    if let Some(entry) =
                        crate::utils::completion_cache::claude_completion_entry(&json_val, &model_name, now)
                    {
                        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
                    }
                }
                Response::builder()
                    .header("Content-Type", "application/json")
                    .status(StatusCode::OK)
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

fn handle_codex_messages_stream(
    stream_res: CodexStreamResult,
    model: &str,
    cache_key: &str,
    show_thinking: bool,
) -> Response {
    let mut sse_state = AnthropicStreamState::new(&format!("msg_{}", uuid::Uuid::new_v4()), model);
    sse_state.show_thinking = show_thinking;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let cache_key = cache_key.to_string();
    stream_sse_transformed(
        stream_res.response.bytes_stream(),
        sse_state,
        move |line, state: &mut AnthropicStreamState| {
            if let Some(parsed) = parse_codex_sse_line(line) {
                codex_to_anthropic_sse(&parsed, state)
            } else {
                Vec::new()
            }
        },
        Some(Box::new(move |state: AnthropicStreamState| {
            if let Some(entry) = crate::utils::completion_cache::codex_completion_entry(
                &state.full_text,
                &state.codex,
                &state.model,
                now,
                state.finished,
            ) {
                crate::utils::completion_cache::write_completion_cache(&cache_key, entry);
            }
        })),
    )
}

async fn handle_codex_messages_buffered(
    stream_res: CodexStreamResult,
    model: &str,
    cache_key: &str,
    show_thinking: bool,
) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_codex_sse_line).collect();
    let collected = collect_codex_response(&chunks);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Some(entry) = crate::utils::completion_cache::codex_completion_entry(
        &collected.text,
        &collected.acc,
        model,
        now,
        collected.finished,
    ) {
        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
    }

    let token = if collected.acc.encrypted.is_empty() {
        String::new()
    } else {
        mint_reasoning_token(
            &serde_json::to_string(&collected.acc.encrypted).unwrap_or_default(),
            &collected.acc.reasoning,
        )
    };

    let mut content = Vec::new();
    if show_thinking && !collected.acc.reasoning.is_empty() {
        let mut thinking_block = serde_json::json!({
            "type": "thinking",
            "thinking": collected.acc.reasoning,
        });
        if !token.is_empty() {
            thinking_block["signature"] = Value::String(token.clone());
        }
        content.push(thinking_block);
    }
    if !collected.text.is_empty() {
        content.push(serde_json::json!({ "type": "text", "text": collected.text }));
    }
    for tool in &collected.acc.tools {
        let parsed: Value = serde_json::from_str(&tool.args).unwrap_or(serde_json::json!({}));
        content.push(serde_json::json!({
            "type": "tool_use",
            "id": tool.id,
            "name": tool.name,
            "input": parsed,
        }));
    }

    let stop_reason = if collected.acc.tools.is_empty() {
        "end_turn"
    } else {
        "tool_use"
    };
    let usage = crate::utils::completion_cache::codex_usage(collected.acc.usage.as_ref());

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": serde_json::Value::Null,
            "usage": crate::utils::completion_cache::anthropic_usage_value(&usage)
        })),
    )
        .into_response()
}

async fn dispatch_codex_messages(
    state: &AppState,
    req: &MessagesRequest,
    upstream_model: &str,
    model_class: ModelClass,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    cache_key: &str,
) -> Response {
    let raw_val = serde_json::to_value(req).unwrap_or_default();
    let responses_body = match map_anthropic_to_responses(&raw_val, Some(upstream_model)) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": e.to_string(), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    let conversation_id = {
        let mut guard = state.pool.write().await;
        let cid = if lookup_keys.is_empty() {
            None
        } else {
            guard
                .resolve_conversation_id(Provider::Codex, lookup_keys)
                .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
        };
        if let Some(ref id) = cid {
            guard.record_conversation_id(Provider::Codex, lookup_keys, Some(id));
        }
        cid
    };

    let params = CallCodexParams {
        body: responses_body,
        model_class,
        preferred_account_id,
        conversation_id,
    };

    let model_name = req.model.as_deref().unwrap_or(upstream_model);
    match call_codex_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Codex, lookup_keys, Some(&stream_res.account_id));
            }
            let show_thinking = req
                .thinking
                .as_ref()
                .is_some_and(|t| t.get("type").and_then(Value::as_str) != Some("disabled"));
            if req.stream.unwrap_or(false) {
                handle_codex_messages_stream(stream_res, model_name, cache_key, show_thinking)
            } else {
                handle_codex_messages_buffered(stream_res, model_name, cache_key, show_thinking).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

fn invalid_body_response() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": { "message": "Invalid JSON request body", "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

fn cache_bypassed(headers: &HeaderMap) -> bool {
    headers
        .get("cache-control")
        .is_some_and(|v| v.as_bytes() == b"no-cache")
}

struct ResolvedRoute {
    candidates: Result<Vec<RouteCandidate>, ModelNotFoundError>,
    cache_key: String,
}

fn resolve_with_cache_key(raw: &Value, model: &str, endpoint: MappingEndpoint, effort: Option<&str>) -> ResolvedRoute {
    let candidates = resolve_route_candidates(model, endpoint, effort);
    let fingerprint = candidates
        .as_ref()
        .ok()
        .and_then(|c| get_route_fingerprint(c, endpoint));
    ResolvedRoute {
        candidates,
        cache_key: completion_key(raw, fingerprint.as_deref()),
    }
}

fn sse_hit_response(events: &[String]) -> Response {
    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("X-Cache", "HIT")
        .body(axum::body::Body::from(events.concat()))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}

fn json_hit_response(value: &Value) -> Response {
    Response::builder()
        .header("Content-Type", "application/json")
        .header("X-Cache", "HIT")
        .status(StatusCode::OK)
        .body(axum::body::Body::from(value.to_string()))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response())
}

fn messages_hit_response(req: &MessagesRequest, entry: &crate::utils::completion_cache::CompletionEntry) -> Response {
    let excluded = reasoning_excluded(req.reasoning.as_ref());
    if req.stream.unwrap_or(false) {
        sse_hit_response(&replay_messages_stream(entry, excluded))
    } else {
        json_hit_response(&replay_messages_json(entry, excluded))
    }
}

fn completions_hit_response(
    raw: &Value,
    req: &CompletionsRequest,
    entry: &crate::utils::completion_cache::CompletionEntry,
) -> Response {
    let excluded = reasoning_excluded(req.reasoning.as_ref());
    let format = reasoning_format_for(req.model.as_deref().unwrap_or(""));
    if req.stream.unwrap_or(false) {
        let want_usage = raw
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        sse_hit_response(&replay_completions_stream(entry, format, excluded, want_usage))
    } else {
        json_hit_response(&replay_completions_json(entry, format, excluded))
    }
}

pub async fn v1_messages_handler(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let raw: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return invalid_body_response(),
    };
    let req: MessagesRequest = match serde_json::from_value(raw.clone()) {
        Ok(r) => r,
        Err(_) => return invalid_body_response(),
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

    tracing::debug!("--> POST /v1/messages (model: {model})");

    let effort = requested_effort(
        req.effort.as_deref(),
        req.reasoning_effort.as_deref(),
        req.reasoning.as_ref(),
        req.output_config.as_ref(),
    );
    let resolved = resolve_with_cache_key(&raw, &model, MappingEndpoint::Messages, effort.as_deref());
    if !cache_bypassed(&headers)
        && let Some(entry) = lookup_completion(&resolved.cache_key)
    {
        tracing::info!("[Cache] HIT for Anthropic Messages: {}", resolved.cache_key);
        return messages_hit_response(&req, &entry);
    }
    let cache_key = resolved.cache_key;
    let candidates = match resolved.candidates {
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

    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();
    let (preferred_account_id, lookup_keys) = {
        let mut guard = state.pool.write().await;
        let pref = guard.resolve_preferred_account(candidate.route.provider, &headers, &messages_val);
        (pref.account_id, pref.lookup_keys)
    };

    match candidate.route.provider {
        Provider::Gemini => {
            dispatch_gemini_messages(
                &state,
                &req,
                &candidate.route,
                preferred_account_id,
                &lookup_keys,
                &cache_key,
            )
            .await
        }
        Provider::Claude => {
            dispatch_claude_messages(
                &state,
                &req,
                &candidate.route,
                preferred_account_id,
                &lookup_keys,
                headers,
                &cache_key,
            )
            .await
        }
        Provider::Codex => {
            dispatch_codex_messages(
                &state,
                &req,
                &candidate.route.upstream_model,
                candidate.route.model_class,
                preferred_account_id,
                &lookup_keys,
                &cache_key,
            )
            .await
        }
    }
}

fn handle_gemini_completions_stream(
    stream_res: GeminiStreamResult,
    model: &str,
    history_key: &str,
    reasoning_excluded: bool,
    cache_key: &str,
    want_usage: bool,
) -> Response {
    let mut sse_state = OpenAiStreamState::new(&format!("chatcmpl-{}", uuid::Uuid::new_v4()), model);
    sse_state.history_key = history_key.to_string();
    sse_state.reasoning_excluded = reasoning_excluded;
    sse_state.want_usage = want_usage;
    let cache_key = cache_key.to_string();
    stream_sse_transformed(
        stream_res.response.bytes_stream(),
        sse_state,
        move |line, state: &mut OpenAiStreamState| {
            if let Some(parsed) = parse_gemini_sse_line(line) {
                gemini_to_openai_sse(&parsed, state)
            } else {
                Vec::new()
            }
        },
        Some(Box::new(move |state: OpenAiStreamState| {
            if let Some(entry) = crate::utils::completion_cache::gemini_stream_entry_from_openai(&state) {
                crate::utils::completion_cache::write_completion_cache(&cache_key, entry);
            }
        })),
    )
}

async fn handle_gemini_completions_buffered(
    stream_res: GeminiStreamResult,
    model: &str,
    history_key: &str,
    reasoning_excluded: bool,
    cache_key: &str,
) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_gemini_sse_line).collect();
    let res = collect_gemini_response(&chunks);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Some(sig) = res.thought_signature.as_deref() {
        record_history(history_key, sig, &res.reasoning);
    }
    if let Some(entry) = crate::utils::completion_cache::gemini_completion_entry(&res, model, now) {
        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
    }

    let mut message = serde_json::json!({ "role": "assistant", "content": res.text });
    if !res.reasoning.is_empty() && !reasoning_excluded {
        message["reasoning"] = Value::String(res.reasoning.clone());
        message["reasoning_content"] = Value::String(res.reasoning.clone());
    }
    let token = mint_reasoning_token(res.thought_signature.as_deref().unwrap_or(""), &res.reasoning);
    if (!res.reasoning.is_empty() || !token.is_empty()) && !reasoning_excluded {
        let mut entry = serde_json::json!({
            "type": "reasoning.text",
            "id": Value::Null,
            "text": res.reasoning,
            "format": reasoning_format_for(model),
            "index": 0
        });
        if !token.is_empty() {
            entry["signature"] = Value::String(token);
        }
        message["reasoning_details"] = Value::Array(vec![entry]);
    }
    let finish_reason = crate::utils::completion_cache::map_gemini_to_openai_finish(
        res.finish_reason.as_deref(),
        !res.function_calls.is_empty(),
    );
    if !res.function_calls.is_empty() {
        let tool_calls: Vec<Value> = res
            .function_calls
            .iter()
            .map(|fc| {
                let tool_id = format!("call_{}", uuid::Uuid::new_v4().simple());
                if let Some(sig) = res.thought_signature.as_deref() {
                    record_tool_signature(&tool_id, sig, "");
                }
                serde_json::json!({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": serde_json::to_string(&fc.args).unwrap_or_default(),
                    }
                })
            })
            .collect();
        message["tool_calls"] = Value::Array(tool_calls);
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.output_tokens,
                "total_tokens": res.prompt_tokens + res.output_tokens,
                "prompt_tokens_details": { "cached_tokens": 0, "audio_tokens": 0 },
                "completion_tokens_details": {
                    "reasoning_tokens": res.reasoning_tokens,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        })),
    )
        .into_response()
}

async fn dispatch_gemini_completions(
    state: &AppState,
    req: &CompletionsRequest,
    route: &RouteDecision,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    cache_key: &str,
) -> Response {
    let is_thinking_requested = requested_effort(
        req.effort.as_deref(),
        req.reasoning_effort.as_deref(),
        req.reasoning.as_ref(),
        req.output_config.as_ref(),
    )
    .is_some()
        || crate::config::model_supports_thinking(&route.upstream_model);
    let raw_val = serde_json::to_value(req).unwrap_or_default();
    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();

    let history_ctx = crate::utils::history_cache::HistoryContext {
        model: req.model.clone().unwrap_or_default(),
        system: Value::Null,
        tools: serde_json::to_value(&req.tools).unwrap_or(Value::Null),
    };
    let (system_instruction, contents) =
        match map_messages_to_gemini(&messages_val, None, is_thinking_requested, Some(&history_ctx)) {
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

    let generation_config = build_generation_config(&raw_val, &route.upstream_model, is_thinking_requested);

    let params = CallGeminiParams {
        contents: contents
            .into_iter()
            .map(|c| serde_json::to_value(c).unwrap_or_default())
            .collect(),
        system_instruction: system_instruction.map(|s| serde_json::to_value(s).unwrap_or_default()),
        generation_config,
        tools: req.tools.as_deref().and_then(map_openai_tools_for_gemini),
        tool_config: None,
        upstream_model: route.upstream_model.clone(),
        model_class: route.model_class,
        preferred_account_id,
    };

    let model_name = req.model.as_deref().unwrap_or(&route.upstream_model);
    let history_key = gemini_history_key(
        req.model.as_deref(),
        &messages_val,
        &Value::Null,
        &serde_json::to_value(&req.tools).unwrap_or(Value::Null),
    );
    let excluded = reasoning_excluded(req.reasoning.as_ref());
    match call_gemini_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Gemini, lookup_keys, Some(&stream_res.account_id));
            }
            if req.stream.unwrap_or(false) {
                let want_usage = req
                    .stream_options
                    .as_ref()
                    .and_then(|o| o.get("include_usage"))
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                handle_gemini_completions_stream(stream_res, model_name, &history_key, excluded, cache_key, want_usage)
            } else {
                handle_gemini_completions_buffered(stream_res, model_name, &history_key, excluded, cache_key).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

fn map_openai_request_to_claude(req: &CompletionsRequest, upstream_model: &str) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    let mut system_text: Option<String> = None;

    if let Some(msgs) = &req.messages {
        for m in msgs {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            if role == "system" || role == "developer" {
                if let Some(s) = m.get("content").and_then(Value::as_str) {
                    system_text = Some(s.to_string());
                }
                continue;
            }
            let content = m
                .get("content")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            messages.push(serde_json::json!({
                "role": if role == "assistant" { "assistant" } else { "user" },
                "content": content,
            }));
        }
    }

    let mapped_model = map_anthropic_model(upstream_model);
    let sys_val = system_text.map(Value::String);
    let system = with_claude_code_identity(sys_val.as_ref());

    let mut out = serde_json::json!({
        "model": mapped_model,
        "messages": messages,
        "system": system,
        "max_tokens": req.max_tokens.unwrap_or(4096),
        "stream": req.stream.unwrap_or(false),
    });

    if let Some(tools) = &req.tools {
        let mapped_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                if let (Some("function"), Some(f)) = (t.get("type").and_then(Value::as_str), t.get("function")) {
                    return serde_json::json!({
                        "name": f.get("name").and_then(Value::as_str).unwrap_or(""),
                        "description": f.get("description").and_then(Value::as_str).unwrap_or(""),
                        "input_schema": f.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({}))
                    });
                }
                t.clone()
            })
            .collect();
        out["tools"] = Value::Array(mapped_tools);
    }

    apply_thinking_constraints(&mut out);
    out
}

fn handle_claude_completions_stream(stream_res: ClaudeStreamResult, model: &str) -> Response {
    let sse_state = OpenAiStreamState::new(&format!("chatcmpl-{}", uuid::Uuid::new_v4()), model);
    stream_sse_transformed(
        stream_res.response.bytes_stream(),
        sse_state,
        move |line, state: &mut OpenAiStreamState| {
            if let Some(parsed) = parse_anthropic_sse_line(line) {
                anthropic_to_openai_sse(&parsed, state)
            } else {
                Vec::new()
            }
        },
        None,
    )
}

async fn handle_claude_completions_buffered(stream_res: ClaudeStreamResult, model: &str, cache_key: &str) -> Response {
    let json_val: Value = stream_res.response.json().await.unwrap_or_default();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Some(entry) = crate::utils::completion_cache::claude_completion_entry(&json_val, model, now) {
        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
    }
    let text = json_val
        .get("content")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let prompt_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
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
                "message": { "role": "assistant", "content": text },
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

async fn dispatch_claude_completions(
    state: &AppState,
    req: &CompletionsRequest,
    route: &RouteDecision,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    headers: HeaderMap,
    cache_key: &str,
) -> Response {
    let claude_body = map_openai_request_to_claude(req, &route.upstream_model);
    let params = CallClaudeParams {
        body: claude_body,
        model_class: route.model_class,
        preferred_account_id,
        client_headers: Some(headers),
    };

    match call_claude_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Claude, lookup_keys, Some(&stream_res.account_id));
            }
            let model_name = req.model.clone().unwrap_or_else(|| route.upstream_model.clone());
            if req.stream.unwrap_or(false) {
                handle_claude_completions_stream(stream_res, &model_name)
            } else {
                handle_claude_completions_buffered(stream_res, &model_name, cache_key).await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

fn handle_codex_completions_stream(
    stream_res: CodexStreamResult,
    model: &str,
    cache_key: &str,
    reasoning_excluded: bool,
    want_usage: bool,
) -> Response {
    let mut sse_state = OpenAiStreamState::new(&format!("chatcmpl-{}", uuid::Uuid::new_v4()), model);
    sse_state.reasoning_excluded = reasoning_excluded;
    sse_state.want_usage = want_usage;
    let cache_key = cache_key.to_string();
    stream_sse_transformed(
        stream_res.response.bytes_stream(),
        sse_state,
        move |line, state: &mut OpenAiStreamState| {
            if let Some(parsed) = parse_codex_sse_line(line) {
                codex_to_openai_sse(&parsed, state)
            } else {
                Vec::new()
            }
        },
        Some(Box::new(move |state: OpenAiStreamState| {
            if let Some(entry) = crate::utils::completion_cache::codex_completion_entry(
                &state.full_text,
                &state.codex,
                &state.model,
                state.created,
                state.finished,
            ) {
                crate::utils::completion_cache::write_completion_cache(&cache_key, entry);
            }
        })),
    )
}

async fn handle_codex_completions_buffered(
    stream_res: CodexStreamResult,
    model: &str,
    cache_key: &str,
    reasoning_excluded: bool,
) -> Response {
    let full_body = stream_res.response.text().await.unwrap_or_default();
    let chunks: Vec<Value> = full_body.lines().filter_map(parse_codex_sse_line).collect();
    let collected = collect_codex_response(&chunks);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Some(entry) = crate::utils::completion_cache::codex_completion_entry(
        &collected.text,
        &collected.acc,
        model,
        now,
        collected.finished,
    ) {
        crate::utils::completion_cache::write_completion_cache(cache_key, entry);
    }

    let token = if collected.acc.encrypted.is_empty() {
        String::new()
    } else {
        mint_reasoning_token(
            &serde_json::to_string(&collected.acc.encrypted).unwrap_or_default(),
            &collected.acc.reasoning,
        )
    };

    let mut message = serde_json::json!({
        "role": "assistant",
        "content": if collected.text.is_empty() { Value::Null } else { Value::String(collected.text) },
    });
    if !reasoning_excluded && !collected.acc.reasoning.is_empty() {
        message["reasoning"] = Value::String(collected.acc.reasoning.clone());
        message["reasoning_content"] = Value::String(collected.acc.reasoning.clone());
    }
    if !reasoning_excluded && (!collected.acc.reasoning.is_empty() || !token.is_empty()) {
        let mut entry = serde_json::json!({
            "type": "reasoning.text",
            "id": Value::Null,
            "text": collected.acc.reasoning,
            "format": reasoning_format_for(model),
            "index": 0,
        });
        if !token.is_empty() {
            entry["signature"] = Value::String(token);
        }
        message["reasoning_details"] = Value::Array(vec![entry]);
    }
    if !collected.acc.tools.is_empty() {
        message["tool_calls"] = Value::Array(
            collected
                .acc
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "id": t.id,
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "arguments": t.args,
                        }
                    })
                })
                .collect(),
        );
    }

    let finish_reason = if collected.acc.tools.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    let usage = crate::utils::completion_cache::codex_usage(collected.acc.usage.as_ref());

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": crate::utils::completion_cache::openai_usage_value(&usage)
        })),
    )
        .into_response()
}

async fn dispatch_codex_completions(
    state: &AppState,
    req: &CompletionsRequest,
    upstream_model: &str,
    model_class: ModelClass,
    preferred_account_id: Option<String>,
    lookup_keys: &[String],
    cache_key: &str,
) -> Response {
    let raw_val = serde_json::to_value(req).unwrap_or_default();
    let responses_body = match map_openai_to_responses(&raw_val, Some(upstream_model)) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": e.to_string(), "type": "invalid_request_error" }
                })),
            )
                .into_response();
        }
    };

    let conversation_id = {
        let mut guard = state.pool.write().await;
        let cid = if lookup_keys.is_empty() {
            None
        } else {
            guard
                .resolve_conversation_id(Provider::Codex, lookup_keys)
                .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
        };
        if let Some(ref id) = cid {
            guard.record_conversation_id(Provider::Codex, lookup_keys, Some(id));
        }
        cid
    };

    let params = CallCodexParams {
        body: responses_body,
        model_class,
        preferred_account_id,
        conversation_id,
    };

    let model_name = req.model.as_deref().unwrap_or(upstream_model);
    match call_codex_stream(&params, &state.pool, &state.http_client).await {
        Ok(stream_res) => {
            {
                let mut guard = state.pool.write().await;
                guard.record_account_pin(Provider::Codex, lookup_keys, Some(&stream_res.account_id));
            }
            if req.stream.unwrap_or(false) {
                let want_usage = req
                    .stream_options
                    .as_ref()
                    .and_then(|o| o.get("include_usage"))
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                handle_codex_completions_stream(
                    stream_res,
                    model_name,
                    cache_key,
                    reasoning_excluded(req.reasoning.as_ref()),
                    want_usage,
                )
            } else {
                handle_codex_completions_buffered(
                    stream_res,
                    model_name,
                    cache_key,
                    reasoning_excluded(req.reasoning.as_ref()),
                )
                .await
            }
        }
        Err(err) => attempt_error_to_response(&err),
    }
}

pub async fn v1_chat_completions_handler(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let raw: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return invalid_body_response(),
    };
    let req: CompletionsRequest = match serde_json::from_value(raw.clone()) {
        Ok(r) => r,
        Err(_) => return invalid_body_response(),
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

    tracing::debug!("--> POST /v1/chat/completions (model: {model})");

    let effort = requested_effort(
        req.effort.as_deref(),
        req.reasoning_effort.as_deref(),
        req.reasoning.as_ref(),
        req.output_config.as_ref(),
    );
    let resolved = resolve_with_cache_key(&raw, &model, MappingEndpoint::OpenAi, effort.as_deref());
    if !cache_bypassed(&headers)
        && let Some(entry) = lookup_completion(&resolved.cache_key)
    {
        tracing::info!("[Cache] HIT for OpenAI Chat completions: {}", resolved.cache_key);
        return completions_hit_response(&raw, &req, &entry);
    }
    let cache_key = resolved.cache_key;
    let candidates = match resolved.candidates {
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

    let messages_val = serde_json::to_value(&req.messages).unwrap_or_default();
    let (preferred_account_id, lookup_keys) = {
        let mut guard = state.pool.write().await;
        let pref = guard.resolve_preferred_account(candidate.route.provider, &headers, &messages_val);
        (pref.account_id, pref.lookup_keys)
    };

    match candidate.route.provider {
        Provider::Gemini => {
            dispatch_gemini_completions(
                &state,
                &req,
                &candidate.route,
                preferred_account_id,
                &lookup_keys,
                &cache_key,
            )
            .await
        }
        Provider::Claude => {
            dispatch_claude_completions(
                &state,
                &req,
                &candidate.route,
                preferred_account_id,
                &lookup_keys,
                headers,
                &cache_key,
            )
            .await
        }
        Provider::Codex => {
            dispatch_codex_completions(
                &state,
                &req,
                &candidate.route.upstream_model,
                candidate.route.model_class,
                preferred_account_id,
                &lookup_keys,
                &cache_key,
            )
            .await
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::app::{AppState, create_app};

    use super::{CompletionsRequest, requested_effort};

    #[test]
    fn test_requested_effort_prefers_explicit_effort() {
        let req: CompletionsRequest = serde_json::from_value(serde_json::json!({
            "model": "x",
            "effort": "low",
            "reasoning_effort": "high",
            "reasoning": { "effort": "max" },
            "output_config": { "effort": "medium" }
        }))
        .unwrap();
        assert_eq!(
            requested_effort(
                req.effort.as_deref(),
                req.reasoning_effort.as_deref(),
                req.reasoning.as_ref(),
                req.output_config.as_ref()
            ),
            Some("low".to_string())
        );
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(
            raw.get("reasoning_effort").and_then(serde_json::Value::as_str),
            Some("high")
        );

        let minimal: CompletionsRequest = serde_json::from_value(serde_json::json!({ "model": "x" })).unwrap();
        assert_eq!(
            requested_effort(
                minimal.effort.as_deref(),
                minimal.reasoning_effort.as_deref(),
                minimal.reasoning.as_ref(),
                minimal.output_config.as_ref()
            ),
            None
        );
        let raw_minimal = serde_json::to_value(&minimal).unwrap();
        assert_eq!(raw_minimal.get("reasoning_effort"), None);
    }

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
    async fn test_completions_hit_serves_cached_stream() {
        use crate::model_mapping::{MappingEndpoint, get_route_fingerprint, resolve_route_candidates};
        use crate::utils::completion_cache::{
            CompletionEntry, CompletionUsage, completion_key, write_completion_cache,
        };
        let state = AppState::default();
        let app = create_app(state);

        let body = serde_json::json!({
            "model": "gemini-3.8-flash-tiered",
            "stream": true,
            "messages": [{ "role": "user", "content": "cache-hit-probe-stream-1" }]
        });
        let candidates = resolve_route_candidates("gemini-3.8-flash-tiered", MappingEndpoint::OpenAi, None).unwrap();
        let fingerprint = get_route_fingerprint(&candidates, MappingEndpoint::OpenAi);
        let key = completion_key(&body, fingerprint.as_deref());
        write_completion_cache(
            &key,
            CompletionEntry {
                text: "cached-stream-reply".to_string(),
                reasoning: None,
                thought_signature: None,
                finish_reason: Some("STOP".to_string()),
                tool_calls: None,
                usage: CompletionUsage {
                    input_tokens: 3,
                    output_tokens: 2,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: 0,
                },
                model: "gemini-3.8-flash-tiered".to_string(),
                created: 0,
            },
        );

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("x-cache").map(|v| v.to_str().unwrap()), Some("HIT"));
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let content: String = text
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .filter_map(|d| serde_json::from_str::<serde_json::Value>(d.trim()).ok())
            .filter_map(|v| {
                v.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(content, "cached-stream-reply");
        assert!(text.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_messages_hit_serves_cached_json() {
        use crate::model_mapping::{MappingEndpoint, get_route_fingerprint, resolve_route_candidates};
        use crate::utils::completion_cache::{
            CompletionEntry, CompletionUsage, completion_key, write_completion_cache,
        };
        let state = AppState::default();
        let app = create_app(state);

        let body = serde_json::json!({
            "model": "gemini-3.8-flash-tiered",
            "max_tokens": 50,
            "messages": [{ "role": "user", "content": "cache-hit-probe-json-2" }]
        });
        let candidates = resolve_route_candidates("gemini-3.8-flash-tiered", MappingEndpoint::Messages, None).unwrap();
        let fingerprint = get_route_fingerprint(&candidates, MappingEndpoint::Messages);
        let key = completion_key(&body, fingerprint.as_deref());
        write_completion_cache(
            &key,
            CompletionEntry {
                text: "cached-json-reply".to_string(),
                reasoning: Some("cached reason".to_string()),
                thought_signature: Some("sig_cached".to_string()),
                finish_reason: Some("STOP".to_string()),
                tool_calls: None,
                usage: CompletionUsage {
                    input_tokens: 4,
                    output_tokens: 3,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: 0,
                },
                model: "gemini-3.8-flash-tiered".to_string(),
                created: 0,
            },
        );

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("x-cache").map(|v| v.to_str().unwrap()), Some("HIT"));
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.get("stop_reason").and_then(|v| v.as_str()), Some("end_turn"));
        let blocks = parsed.get("content").and_then(|v| v.as_array()).unwrap();
        assert_eq!(
            blocks[0].get("thinking").and_then(|v| v.as_str()),
            Some("cached reason")
        );
    }

    #[tokio::test]
    async fn test_stream_end_hook_populates_completion_cache() {
        use crate::utils::completion_cache::{lookup_completion, write_completion_cache};
        use crate::utils::sse::OpenAiStreamState;
        let key = format!("test-stream-hook-{}", std::process::id());
        let mut state = OpenAiStreamState::new("chatcmpl-hook", "hook-model");
        state.finished = true;
        state.full_text = "hooked-reply".to_string();
        state.finish_reason = Some("STOP".to_string());
        let key_owned = key.clone();
        let res = super::stream_sse_transformed(
            futures::stream::iter(Vec::<Result<axum::body::Bytes, reqwest::Error>>::new()),
            state,
            |_line: &str, _state: &mut OpenAiStreamState| Vec::new(),
            Some(Box::new(move |done: OpenAiStreamState| {
                if let Some(entry) = crate::utils::completion_cache::gemini_stream_entry_from_openai(&done) {
                    write_completion_cache(&key_owned, entry);
                }
            })),
        );
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert!(bytes.is_empty());
        let entry = lookup_completion(&key).expect("hook must populate the cache at stream end");
        assert_eq!(entry.text, "hooked-reply");
    }

    #[tokio::test]
    async fn test_legacy_key_hit_served_when_route_unknown() {
        use crate::utils::completion_cache::{
            CompletionEntry, CompletionUsage, completion_key, write_completion_cache,
        };
        let state = AppState::default();
        let app = create_app(state);

        let body = serde_json::json!({
            "model": "unknown-nonexistent-model",
            "messages": [{ "role": "user", "content": "cache-hit-probe-legacy-3" }]
        });
        let key = completion_key(&body, None);
        write_completion_cache(
            &key,
            CompletionEntry {
                text: "cached-legacy-reply".to_string(),
                reasoning: None,
                thought_signature: None,
                finish_reason: Some("STOP".to_string()),
                tool_calls: None,
                usage: CompletionUsage::default(),
                model: "unknown-nonexistent-model".to_string(),
                created: 0,
            },
        );

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("x-cache").map(|v| v.to_str().unwrap()), Some("HIT"));
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("cached-legacy-reply"));
    }

    #[tokio::test]
    async fn test_messages_handler_routes_and_dispatches_gemini() {
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
    async fn test_messages_handler_routes_and_dispatches_claude() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "claude-sonnet-5-native",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_messages_handler_routes_and_dispatches_codex() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/messages")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "gpt-5.5",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_completions_handler_routes_and_dispatches_gemini() {
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

    #[tokio::test]
    async fn test_completions_handler_routes_and_dispatches_claude() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "claude-sonnet-5-native",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_completions_handler_routes_and_dispatches_codex() {
        let state = AppState::default();
        let app = create_app(state);

        let req = Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "gpt-5.5",
                    "messages": [{ "role": "user", "content": "hi" }]
                })
                .to_string(),
            ))
            .unwrap();

        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }
}
