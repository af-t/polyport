use serde_json::Value;

use crate::mappers::media::{MediaError, parse_anthropic_media_block, parse_openai_media_block, render_for_codex};
use crate::services::anthropic::normalize_anthropic_effort;

const EFFORT_TIERS: &[&str] = &["low", "medium", "high", "xhigh", "max", "ultra"];

#[must_use]
pub fn map_effort(model: Option<&str>, effort: Option<&str>) -> String {
    let ceiling_idx = match model {
        Some("gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna" | "gpt-6-luna") => 4, // max
        _ => 3,                                                                     // xhigh
    };

    let requested = match effort {
        Some("minimal") => "low",
        Some(s) => s.trim(),
        None => "medium",
    };

    let lower = requested.to_lowercase();
    let idx = EFFORT_TIERS.iter().position(|&t| t == lower.as_str()).unwrap_or(1); // default "medium"

    EFFORT_TIERS[idx.min(ceiling_idx)].to_string()
}

#[must_use]
pub fn map_codex_finish_to_anthropic_stop(finish: &str) -> &'static str {
    match finish {
        "tool_calls" | "function_call" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|c| match c {
                Value::String(s) => Some(s.as_str()),
                Value::Object(map) => map.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    }
}

fn map_openai_user_content(content: &Value) -> Result<Vec<Value>, MediaError> {
    match content {
        Value::String(s) => Ok(vec![serde_json::json!({ "type": "input_text", "text": s })]),
        Value::Array(arr) => {
            let mut parts = Vec::new();
            for item in arr {
                if item.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = item.get("text").and_then(Value::as_str) {
                        parts.push(serde_json::json!({ "type": "input_text", "text": t }));
                    }
                } else if let Some(media) = parse_openai_media_block(item) {
                    let rendered = render_for_codex(&media)?;
                    parts.push(rendered);
                } else if item
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| matches!(t, "input_text" | "input_image" | "input_file"))
                {
                    parts.push(item.clone());
                }
            }
            Ok(parts)
        }
        _ => Ok(Vec::new()),
    }
}

fn map_openai_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            if t.get("type").and_then(Value::as_str) == Some("function") {
                if let Some(func) = t.get("function") {
                    return serde_json::json!({
                        "type": "function",
                        "name": func.get("name").and_then(Value::as_str).unwrap_or(""),
                        "description": func.get("description").and_then(Value::as_str).unwrap_or(""),
                        "parameters": func.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({})),
                        "strict": func.get("strict").and_then(Value::as_bool).unwrap_or(false)
                    });
                }
                return t.clone();
            }
            t.clone()
        })
        .collect()
}

/// Maps `OpenAI` Chat Completions payload into Codex Responses API format.
///
/// # Errors
/// Fails if media processing encounters an unsupported media type.
pub fn map_openai_to_responses(body: &Value, model_override: Option<&str>) -> Result<Value, MediaError> {
    let upstream_model = model_override
        .or_else(|| body.get("model").and_then(Value::as_str))
        .unwrap_or("gpt-5.5");

    let mut system_instructions = Vec::new();
    let mut input = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        for msg in msgs {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = msg.get("content").unwrap_or(&Value::Null);

            match role {
                "system" | "developer" => {
                    let text = text_from_content(content);
                    if !text.trim().is_empty() {
                        system_instructions.push(text);
                    }
                }
                "user" => {
                    let parts = map_openai_user_content(content)?;
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "user",
                        "content": parts
                    }));
                }
                "assistant" => {
                    let text = text_from_content(content);
                    if !text.is_empty() {
                        input.push(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": text }]
                        }));
                    }
                    if let Some(tcs) = msg.get("tool_calls").and_then(Value::as_array) {
                        for tc in tcs {
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": tc.get("id").and_then(Value::as_str).unwrap_or(""),
                                "name": tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or(""),
                                "arguments": tc.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("")
                            }));
                        }
                    }
                }
                "tool" => {
                    input.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id").and_then(Value::as_str).unwrap_or(""),
                        "output": text_from_content(content)
                    }));
                }
                _ => {}
            }
        }
    }

    let effort_raw = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str)
        .or_else(|| body.get("reasoning_effort").and_then(Value::as_str))
        .or_else(|| {
            body.get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(Value::as_str)
        });

    let mut responses = serde_json::json!({
        "model": upstream_model,
        "stream": true,
        "store": false,
        "input": input,
        "instructions": if system_instructions.is_empty() { "You are a helpful assistant.".to_string() } else { system_instructions.join("\n\n") },
        "include": ["reasoning.encrypted_content"],
        "reasoning": {
            "effort": map_effort(Some(upstream_model), effort_raw),
            "summary": "auto"
        }
    });

    let tools_opt = body.get("tools").and_then(Value::as_array).filter(|t| !t.is_empty());
    if let Some(tools) = tools_opt {
        responses["tools"] = Value::Array(map_openai_tools(tools));
        let parallel = body.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true);
        responses["parallel_tool_calls"] = Value::Bool(parallel);
    }

    if let Some(tc) = body.get("tool_choice") {
        if let Some(s) = tc.as_str() {
            responses["tool_choice"] = Value::String(s.to_string());
        } else if let Some(fname) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) {
            responses["tool_choice"] = serde_json::json!({ "type": "function", "name": fname });
        }
    }

    Ok(responses)
}

fn map_anthropic_user_blocks(blocks: &[Value]) -> Result<(Vec<Value>, Vec<Value>), MediaError> {
    let mut user_parts = Vec::new();
    let mut tool_outputs = Vec::new();

    for block in blocks {
        let btype = block.get("type").and_then(Value::as_str).unwrap_or("");
        if btype == "text" {
            if let Some(t) = block.get("text").and_then(Value::as_str) {
                user_parts.push(serde_json::json!({ "type": "input_text", "text": t }));
            }
        } else if btype == "tool_result" {
            tool_outputs.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                "output": text_from_content(block.get("content").unwrap_or(&Value::Null))
            }));
        } else if let Some(media) = parse_anthropic_media_block(block) {
            let rendered = render_for_codex(&media)?;
            user_parts.push(rendered);
        }
    }

    Ok((user_parts, tool_outputs))
}

fn extract_anthropic_assistant_input(blocks: &[Value]) -> Vec<Value> {
    let mut items = Vec::new();
    let mut text = String::new();
    for b in blocks {
        if let (Some("text"), Some(t)) = (
            b.get("type").and_then(Value::as_str),
            b.get("text").and_then(Value::as_str),
        ) {
            text.push_str(t);
        }
    }
    if !text.is_empty() {
        items.push(serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text }]
        }));
    }

    for b in blocks {
        if b.get("type").and_then(Value::as_str) == Some("tool_use") {
            let args = b.get("input").map_or_else(|| "{}".to_string(), ToString::to_string);
            items.push(serde_json::json!({
                "type": "function_call",
                "call_id": b.get("id").and_then(Value::as_str).unwrap_or(""),
                "name": b.get("name").and_then(Value::as_str).unwrap_or(""),
                "arguments": args
            }));
        }
    }
    items
}

fn extract_anthropic_instructions(system: Option<&Value>) -> String {
    match system {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|b| match b {
                Value::String(s) => Some(s.as_str()),
                Value::Object(map) => map.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => "You are a helpful assistant.".to_string(),
    }
}

fn extract_anthropic_tools(body: &Value) -> Option<(Vec<Value>, bool)> {
    let tools = body.get("tools").and_then(Value::as_array)?;
    if tools.is_empty() {
        return None;
    }
    let mapped: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "name": t.get("name").and_then(Value::as_str).unwrap_or(""),
                "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                "parameters": t.get("input_schema").cloned().unwrap_or_else(|| serde_json::json!({})),
                "strict": false
            })
        })
        .collect();
    let disable_parallel = body
        .get("tool_choice")
        .and_then(|tc| tc.get("disable_parallel_tool_use"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Some((mapped, !disable_parallel))
}

/// Maps Anthropic Messages payload into Codex Responses API format.
///
/// # Errors
/// Fails if media processing encounters an unsupported media type.
pub fn map_anthropic_to_responses(body: &Value, model_override: Option<&str>) -> Result<Value, MediaError> {
    let upstream_model = model_override
        .or_else(|| body.get("model").and_then(Value::as_str))
        .unwrap_or("gpt-5.5");

    let mut input = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        for msg in msgs {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = msg.get("content").unwrap_or(&Value::Null);

            if role == "system" || role == "developer" {
                let text = text_from_content(content);
                if !text.trim().is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }]
                    }));
                }
                continue;
            }

            let blocks = match content {
                Value::String(s) => vec![serde_json::json!({ "type": "text", "text": s })],
                Value::Array(arr) => arr.clone(),
                _ => Vec::new(),
            };

            if role == "user" {
                let (user_parts, tool_outputs) = map_anthropic_user_blocks(&blocks)?;
                for out in tool_outputs {
                    input.push(out);
                }
                if !user_parts.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "user",
                        "content": user_parts
                    }));
                }
            } else if role == "assistant" {
                input.extend(extract_anthropic_assistant_input(&blocks));
            }
        }
    }

    let instructions = extract_anthropic_instructions(body.get("system"));

    let effort_raw = normalize_anthropic_effort(
        body.get("output_config")
            .and_then(|o| o.get("effort"))
            .and_then(Value::as_str),
    )
    .or_else(|| {
        body.get("reasoning_effort")
            .and_then(Value::as_str)
            .map(ToString::to_string)
    })
    .or_else(|| {
        body.get("reasoning")
            .and_then(|r| r.get("effort"))
            .and_then(Value::as_str)
            .map(ToString::to_string)
    });

    let mut responses = serde_json::json!({
        "model": upstream_model,
        "stream": true,
        "store": false,
        "input": input,
        "instructions": instructions,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {
            "effort": map_effort(Some(upstream_model), effort_raw.as_deref()),
            "summary": "concise"
        }
    });

    if let Some((mapped_tools, parallel)) = extract_anthropic_tools(body) {
        responses["tools"] = Value::Array(mapped_tools);
        responses["parallel_tool_calls"] = Value::Bool(parallel);
    }

    Ok(responses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_map_effort_levels_and_ceiling() {
        assert_eq!(map_effort(Some("gpt-5.5"), Some("minimal")), "low");
        assert_eq!(map_effort(Some("gpt-5.5"), Some("high")), "high");
        assert_eq!(map_effort(Some("gpt-5.5"), Some("max")), "xhigh"); // capped at xhigh
        assert_eq!(map_effort(Some("gpt-5.6-sol"), Some("max")), "max"); // sol allows max
        assert_eq!(map_effort(None, None), "medium");
    }

    #[test]
    fn test_map_codex_finish_to_anthropic_stop() {
        assert_eq!(map_codex_finish_to_anthropic_stop("stop"), "end_turn");
        assert_eq!(map_codex_finish_to_anthropic_stop("complete"), "end_turn");
        assert_eq!(map_codex_finish_to_anthropic_stop("tool_calls"), "tool_use");
        assert_eq!(map_codex_finish_to_anthropic_stop("length"), "max_tokens");
    }

    #[test]
    fn test_map_openai_to_responses_payload() {
        let req = json!({
            "model": "gpt-5.5",
            "messages": [
                { "role": "system", "content": "You are Codex." },
                { "role": "user", "content": "Hello" }
            ],
            "reasoning_effort": "high"
        });

        let resp = map_openai_to_responses(&req, None).unwrap();
        assert_eq!(resp["model"], "gpt-5.5");
        assert_eq!(resp["instructions"], "You are Codex.");
        assert_eq!(resp["reasoning"]["effort"], "high");
        let input = resp["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn test_map_anthropic_to_responses_payload() {
        let req = json!({
            "model": "gpt-5.5",
            "system": "You are Anthropic assistant.",
            "messages": [
                { "role": "user", "content": "Hi there" }
            ]
        });

        let resp = map_anthropic_to_responses(&req, None).unwrap();
        assert_eq!(resp["model"], "gpt-5.5");
        assert_eq!(resp["instructions"], "You are Anthropic assistant.");
        let input = resp["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["text"], "Hi there");
    }
}
