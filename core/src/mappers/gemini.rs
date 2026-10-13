use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mappers::media::{MediaError, parse_anthropic_media_block, parse_openai_media_block, render_for_gemini};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiSystemInstruction {
    pub parts: Vec<Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum MapperError {
    #[error("media mapping error: {0}")]
    Media(#[from] MediaError),
    #[error("invalid message format: {0}")]
    InvalidFormat(String),
}

#[must_use]
pub fn parse_function_response(content: &Value) -> Value {
    if content.is_null() {
        return serde_json::json!({});
    }

    if let Some(s) = content.as_str() {
        if let Ok(parsed) = serde_json::from_str::<Value>(s)
            && parsed.is_object()
        {
            return parsed;
        }
        return serde_json::json!({ "result": s });
    }

    if content.is_object() {
        content.clone()
    } else {
        serde_json::json!({ "result": content })
    }
}

/// Maps content (string or array of blocks) into Gemini parts.
///
/// # Errors
/// Returns `MapperError` if any media item cannot be rendered.
pub fn map_content_parts(content: &Value, is_thinking_requested: bool) -> Result<Vec<Value>, MapperError> {
    if let Some(s) = content.as_str() {
        return Ok(vec![serde_json::json!({ "text": s })]);
    }

    let Some(arr) = content.as_array() else {
        return Ok(Vec::new());
    };

    let mut parts = Vec::new();

    for block in arr {
        if let Some(media) = parse_openai_media_block(block).or_else(|| parse_anthropic_media_block(block)) {
            parts.push(render_for_gemini(&media)?);
            continue;
        }

        let Some(block_type) = block.get("type").and_then(Value::as_str) else {
            continue;
        };

        match block_type {
            "text" => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                parts.push(serde_json::json!({ "text": text }));
            }
            "thinking" => {
                if is_thinking_requested {
                    let sig = block
                        .get("signature")
                        .or_else(|| block.get("thoughtSignature"))
                        .or_else(|| block.get("thought_signature"))
                        .and_then(Value::as_str);

                    if let Some(sig_str) = sig
                        && sig_str != "dummy_sig_from_proxy"
                    {
                        let thinking_text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                        parts.push(serde_json::json!({
                            "text": thinking_text,
                            "thought": true,
                            "thoughtSignature": sig_str,
                            "thought_signature": sig_str
                        }));
                    }
                }
            }
            "tool_use" => {
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let args = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                let id = block.get("id").and_then(Value::as_str);
                parts.push(serde_json::json!({
                    "functionCall": {
                        "name": name,
                        "args": args,
                        "id": id
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block.get("tool_use_id").and_then(Value::as_str);
                let name = block.get("name").and_then(Value::as_str);
                let raw_content = block.get("content").unwrap_or(&Value::Null);
                let resp = parse_function_response(raw_content);
                parts.push(serde_json::json!({
                    "functionResponse": {
                        "name": name,
                        "response": resp,
                        "id": tool_use_id
                    }
                }));
            }
            _ => {}
        }
    }

    Ok(parts)
}

/// Maps client messages to Gemini contents and system instruction.
///
/// # Errors
/// Returns `MapperError` on media translation failure.
pub fn map_messages_to_gemini(
    messages: &Value,
    system_prompt: Option<&str>,
    is_thinking_requested: bool,
) -> Result<(Option<GeminiSystemInstruction>, Vec<GeminiContent>), MapperError> {
    let mut system_texts = Vec::new();
    if let Some(sp) = system_prompt
        && !sp.is_empty()
    {
        system_texts.push(sp.to_string());
    }

    let arr = messages
        .as_array()
        .ok_or_else(|| MapperError::InvalidFormat("messages must be an array".to_string()))?;

    let mut contents = Vec::new();

    for m in arr {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content_val = m.get("content").unwrap_or(&Value::Null);

        if role == "system" {
            if let Some(txt) = content_val.as_str() {
                system_texts.push(txt.to_string());
            } else if let Some(sub_arr) = content_val.as_array() {
                for block in sub_arr {
                    if let Some(txt) = block.get("text").and_then(Value::as_str) {
                        system_texts.push(txt.to_string());
                    }
                }
            }
            continue;
        }

        if role == "tool" {
            let tool_call_id = m.get("tool_call_id").and_then(Value::as_str);
            let name = m.get("name").and_then(Value::as_str);
            let resp = parse_function_response(content_val);
            contents.push(GeminiContent {
                role: "user".to_string(),
                parts: vec![serde_json::json!({
                    "functionResponse": {
                        "name": name,
                        "response": resp,
                        "id": tool_call_id
                    }
                })],
            });
            continue;
        }

        let is_assistant = role == "assistant";
        let target_role = if is_assistant { "model" } else { "user" };
        let mut parts = map_content_parts(content_val, is_thinking_requested)?;

        if is_assistant && let Some(tool_calls) = m.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let function = tc.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let raw_args = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let args = serde_json::from_str::<Value>(raw_args).unwrap_or(serde_json::json!({}));
                let id = tc.get("id").and_then(Value::as_str);
                parts.push(serde_json::json!({
                    "functionCall": {
                        "name": name,
                        "args": args,
                        "id": id
                    }
                }));
            }
        }

        contents.push(GeminiContent {
            role: target_role.to_string(),
            parts,
        });
    }

    let system_instruction = if system_texts.is_empty() {
        None
    } else {
        Some(GeminiSystemInstruction {
            parts: vec![serde_json::json!({ "text": system_texts.join("\n\n") })],
        })
    };

    Ok((system_instruction, contents))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function_response_normalizes_to_object() {
        assert_eq!(parse_function_response(&serde_json::Value::Null), serde_json::json!({}));

        assert_eq!(
            parse_function_response(&serde_json::json!("raw text string")),
            serde_json::json!({ "result": "raw text string" })
        );

        assert_eq!(
            parse_function_response(&serde_json::json!("{\"key\":\"val\"}")),
            serde_json::json!({ "key": "val" })
        );

        assert_eq!(
            parse_function_response(&serde_json::json!([1, 2, 3])),
            serde_json::json!({ "result": [1, 2, 3] })
        );
    }

    #[test]
    fn test_map_content_parts_string_and_blocks() {
        let string_content = serde_json::json!("hello world");
        let parts = map_content_parts(&string_content, false).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].get("text").and_then(serde_json::Value::as_str),
            Some("hello world")
        );

        let blocks_content = serde_json::json!([
            { "type": "text", "text": "foo" },
            { "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } }
        ]);
        let parts2 = map_content_parts(&blocks_content, false).unwrap();
        assert_eq!(parts2.len(), 2);
        assert_eq!(parts2[0].get("text").and_then(serde_json::Value::as_str), Some("foo"));
        assert!(parts2[1].get("functionCall").is_some());
    }

    #[test]
    fn test_map_content_parts_thinking_block_preserves_signature() {
        let blocks = serde_json::json!([
            {
                "type": "thinking",
                "thinking": "thought process",
                "signature": "sig_123"
            }
        ]);
        let parts = map_content_parts(&blocks, true).unwrap();
        assert_eq!(parts.len(), 1);
        let part = &parts[0];
        assert_eq!(
            part.get("text").and_then(serde_json::Value::as_str),
            Some("thought process")
        );
        assert_eq!(part.get("thought").and_then(serde_json::Value::as_bool), Some(true));
        assert_eq!(
            part.get("thoughtSignature").and_then(serde_json::Value::as_str),
            Some("sig_123")
        );
    }

    #[test]
    fn test_map_messages_to_gemini_extracts_system_and_user_turns() {
        let messages = serde_json::json!([
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": "Hello!" }
        ]);
        let (system_instruction, contents) = map_messages_to_gemini(&messages, None, false).unwrap();
        assert!(system_instruction.is_some());
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[1].role, "model");
    }
}
