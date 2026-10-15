use axum::Json;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::get_models_catalog;
use crate::services::credentials::{get_claude_accounts, get_codex_accounts};

fn model_list_entry(entry: &crate::models::ModelDescriptor) -> Value {
    let mut obj = json!({
        "id": entry.id,
        "object": "model",
        "created": created_ts(),
        "owned_by": owned_by(&entry.provider),
        "name": entry.name,
        "featured": entry.featured,
        "thinking": entry.thinking,
        "effort": entry.effort,
    });
    if let Some(modalities) = &entry.modalities {
        obj["modalities"] = json!({"input": modalities.input, "output": modalities.output});
    }
    if let Some(context_window) = entry.context_window {
        obj["context_window"] = json!(context_window);
    }
    if let Some(max_output) = entry.max_output {
        obj["max_output"] = json!(max_output);
    }
    obj
}

fn model_list_visible(entry: &crate::models::ModelDescriptor) -> bool {
    if entry.provider == "anthropic" && get_claude_accounts().is_empty() {
        return false;
    }
    if entry.provider == "codex" && get_codex_accounts().is_empty() {
        return false;
    }
    true
}

pub async fn models_handler() -> Response {
    let Ok(catalog) = get_models_catalog() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response();
    };
    let data: Vec<Value> = catalog
        .iter()
        .filter(|m| model_list_visible(m))
        .map(model_list_entry)
        .collect();
    (StatusCode::OK, Json(json!({ "object": "list", "data": data }))).into_response()
}

fn owned_by(provider: &str) -> &str {
    match provider {
        "google" => "google",
        "anthropic" => "anthropic",
        "codex" => "openai",
        _ => provider,
    }
}

fn created_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
        .saturating_sub(3600 * 24)
}

pub async fn model_retrieve_handler(Path(model_id): Path<String>) -> Response {
    let Ok(catalog) = get_models_catalog() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response();
    };
    let entry = catalog.iter().find(|m| m.id == model_id);
    let Some(entry) = entry else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("The model '{model_id}' does not exist or you do not have access to it."),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response();
    };
    if entry.provider == "anthropic" && get_claude_accounts().is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("The model '{model_id}' does not exist or you do not have access to it."),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response();
    }
    if entry.provider == "codex" && get_codex_accounts().is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("The model '{model_id}' does not exist or you do not have access to it."),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response();
    }
    let mut obj = json!({
        "id": entry.id,
        "object": "model",
        "created": created_ts(),
        "owned_by": owned_by(&entry.provider),
        "name": entry.name,
        "featured": entry.featured,
        "thinking": entry.thinking,
        "effort": entry.effort,
    });
    if let Some(modalities) = &entry.modalities {
        obj["modalities"] = json!({"input": modalities.input, "output": modalities.output});
    }
    (StatusCode::OK, Json::<Value>(obj)).into_response()
}
