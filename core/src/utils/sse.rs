use serde_json::Value;

#[must_use]
pub fn parse_gemini_sse_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data: ") {
        return None;
    }
    let data = trimmed[6..].trim();
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str::<Value>(data).ok()
}

#[must_use]
pub fn parse_anthropic_sse_line(line: &str) -> Option<Value> {
    parse_gemini_sse_line(line)
}

pub struct AnthropicStreamState {
    pub msg_id: String,
    pub model: String,
    pub started: bool,
    pub text_block_started: bool,
    pub block_index: usize,
    pub finished: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl AnthropicStreamState {
    #[must_use]
    pub fn new(msg_id: &str, model: &str) -> Self {
        Self {
            msg_id: msg_id.to_string(),
            model: model.to_string(),
            started: false,
            text_block_started: false,
            block_index: 0,
            finished: false,
            input_tokens: 0,
            output_tokens: 0,
        }
    }
}

#[must_use]
pub fn gemini_to_anthropic_sse(chunk: &Value, state: &mut AnthropicStreamState) -> Vec<String> {
    let mut events = Vec::new();

    // Check usage
    if let Some(usage) = chunk.get("response").and_then(|r| r.get("usageMetadata")) {
        if let Some(prompt) = usage.get("promptTokenCount").and_then(Value::as_u64) {
            state.input_tokens = u32::try_from(prompt).unwrap_or(0);
        }
        if let Some(cand) = usage.get("candidatesTokenCount").and_then(Value::as_u64) {
            state.output_tokens = u32::try_from(cand).unwrap_or(0);
        }
    }

    if !state.started {
        state.started = true;
        let start_event = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": &state.msg_id,
                "type": "message",
                "role": "assistant",
                "model": &state.model,
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": {
                    "input_tokens": state.input_tokens,
                    "output_tokens": 0
                }
            }
        });
        events.push(format!("event: message_start\ndata: {start_event}\n\n"));
    }

    let candidate = chunk
        .get("response")
        .and_then(|r| r.get("candidates"))
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let parts = candidate
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    if let Some(parts_arr) = parts {
        for part in parts_arr {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !state.text_block_started {
                    state.text_block_started = true;
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": state.block_index,
                        "content_block": {
                            "type": "text",
                            "text": ""
                        }
                    });
                    events.push(format!("event: content_block_start\ndata: {block_start}\n\n"));
                }

                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": state.block_index,
                    "delta": {
                        "type": "text_delta",
                        "text": text
                    }
                });
                events.push(format!("event: content_block_delta\ndata: {delta}\n\n"));
            }
        }
    }

    let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(Value::as_str);
    if (finish_reason.is_some() || chunk.get("response").is_none()) && !state.finished {
        state.finished = true;
        if state.text_block_started {
            let block_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": state.block_index
            });
            events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));
        }

        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": serde_json::Value::Null
            },
            "usage": {
                "output_tokens": state.output_tokens
            }
        });
        events.push(format!("event: message_delta\ndata: {msg_delta}\n\n"));
        events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());
    }

    events
}

pub struct OpenAiStreamState {
    pub completion_id: String,
    pub model: String,
    pub started: bool,
    pub finished: bool,
    pub created: u64,
}

impl OpenAiStreamState {
    #[must_use]
    pub fn new(completion_id: &str, model: &str) -> Self {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            completion_id: completion_id.to_string(),
            model: model.to_string(),
            started: false,
            finished: false,
            created,
        }
    }
}

#[must_use]
pub fn gemini_to_openai_sse(chunk: &Value, state: &mut OpenAiStreamState) -> Vec<String> {
    let mut events = Vec::new();

    if !state.started {
        state.started = true;
        let start_chunk = serde_json::json!({
            "id": &state.completion_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": &state.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": serde_json::Value::Null
            }]
        });
        events.push(format!("data: {start_chunk}\n\n"));
    }

    let candidate = chunk
        .get("response")
        .and_then(|r| r.get("candidates"))
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let parts = candidate
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    if let Some(parts_arr) = parts {
        for part in parts_arr {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let text_chunk = serde_json::json!({
                    "id": &state.completion_id,
                    "object": "chat.completion.chunk",
                    "created": state.created,
                    "model": &state.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": text
                        },
                        "finish_reason": serde_json::Value::Null
                    }]
                });
                events.push(format!("data: {text_chunk}\n\n"));
            }
        }
    }

    let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(Value::as_str);
    if (finish_reason.is_some() || chunk.get("response").is_none()) && !state.finished {
        state.finished = true;
        let final_chunk = serde_json::json!({
            "id": &state.completion_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": &state.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        events.push(format!("data: {final_chunk}\n\n"));
        events.push("data: [DONE]\n\n".to_string());
    }

    events
}

#[must_use]
pub fn anthropic_to_openai_sse(chunk: &Value, state: &mut OpenAiStreamState) -> Vec<String> {
    let mut events = Vec::new();
    if !state.started {
        state.started = true;
        let start_chunk = serde_json::json!({
            "id": &state.completion_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": &state.model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "" },
                "finish_reason": serde_json::Value::Null
            }]
        });
        events.push(format!("data: {start_chunk}\n\n"));
    }

    let chunk_type = chunk.get("type").and_then(Value::as_str).unwrap_or("");
    if chunk_type == "content_block_delta" {
        if let Some(text) = chunk.get("delta").and_then(|d| d.get("text")).and_then(Value::as_str) {
            let text_chunk = serde_json::json!({
                "id": &state.completion_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": &state.model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": text },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            events.push(format!("data: {text_chunk}\n\n"));
        }
    } else if (chunk_type == "message_stop" || chunk_type == "message_delta") && !state.finished {
        let stop_reason = chunk
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(Value::as_str);
        if stop_reason.is_some() || chunk_type == "message_stop" {
            state.finished = true;
            let finish_chunk = serde_json::json!({
                "id": &state.completion_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": &state.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            events.push(format!("data: {finish_chunk}\n\n"));
            events.push("data: [DONE]\n\n".to_string());
        }
    }
    events
}

#[must_use]
pub fn collect_gemini_response(chunks: &[Value]) -> (String, Option<String>, u32, u32) {
    let mut full_text = String::new();
    let mut finish_reason = None;
    let mut prompt_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        if let Some(usage) = chunk.get("response").and_then(|r| r.get("usageMetadata")) {
            if let Some(p) = usage.get("promptTokenCount").and_then(Value::as_u64) {
                prompt_tokens = u32::try_from(p).unwrap_or(0);
            }
            if let Some(c) = usage.get("candidatesTokenCount").and_then(Value::as_u64) {
                output_tokens = u32::try_from(c).unwrap_or(0);
            }
        }

        let candidate = chunk
            .get("response")
            .and_then(|r| r.get("candidates"))
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first());

        if let Some(cand) = candidate {
            if let Some(fr) = cand.get("finishReason").and_then(Value::as_str) {
                finish_reason = Some(fr.to_string());
            }

            if let Some(parts) = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        full_text.push_str(t);
                    }
                }
            }
        }
    }

    (full_text, finish_reason, prompt_tokens, output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gemini_sse_line() {
        let line = "data: {\"response\": {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"hello\"}]}}]}}";
        let parsed = parse_gemini_sse_line(line).unwrap();
        assert!(parsed.get("response").is_some());

        assert!(parse_gemini_sse_line("data: [DONE]").is_none());
        assert!(parse_gemini_sse_line(": ping").is_none());
        assert!(parse_gemini_sse_line("").is_none());
    }

    #[test]
    fn test_gemini_to_anthropic_sse_events() {
        let mut state = AnthropicStreamState::new("test-msg", "gemini-3.5-flash");
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [{ "text": "hello" }] },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2
                }
            }
        });

        let events = gemini_to_anthropic_sse(&chunk, &mut state);
        assert_ne!(events.len(), 0);
        let all_text = events.join("\n");
        assert!(all_text.contains("message_start"));
        assert!(all_text.contains("content_block_delta"));
        assert!(all_text.contains("message_stop"));
    }

    #[test]
    fn test_gemini_to_openai_sse_events() {
        let mut state = OpenAiStreamState::new("test-cmpl", "gemini-3.5-flash");
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [{ "text": "hello world" }] },
                    "finishReason": "STOP"
                }]
            }
        });

        let events = gemini_to_openai_sse(&chunk, &mut state);
        assert_ne!(events.len(), 0);
        let all_text = events.join("\n");
        assert!(all_text.contains("chat.completion.chunk"));
        assert!(all_text.contains("[DONE]"));
    }

    #[test]
    fn test_collect_gemini_response_buffered() {
        let chunk1 = serde_json::json!({
            "response": {
                "candidates": [{ "content": { "parts": [{ "text": "foo " }] } }]
            }
        });
        let chunk2 = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [{ "text": "bar" }] },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2
                }
            }
        });

        let (text, finish_reason, prompt_tokens, output_tokens) = collect_gemini_response(&[chunk1, chunk2]);
        assert_eq!(text, "foo bar");
        assert_eq!(finish_reason.as_deref(), Some("STOP"));
        assert_eq!(prompt_tokens, 5);
        assert_eq!(output_tokens, 2);
    }

    #[test]
    fn test_anthropic_to_openai_sse_events() {
        let mut state = OpenAiStreamState::new("test-cmpl-anthropic", "claude-sonnet-5");
        let delta_chunk = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "hello anthropic"
            }
        });
        let stop_chunk = serde_json::json!({
            "type": "message_stop"
        });

        let events1 = anthropic_to_openai_sse(&delta_chunk, &mut state);
        let events2 = anthropic_to_openai_sse(&stop_chunk, &mut state);

        let all_events = [events1, events2].concat();
        let all_text = all_events.join("\n");
        assert!(all_text.contains("chat.completion.chunk"));
        assert!(all_text.contains("hello anthropic"));
        assert!(all_text.contains("[DONE]"));
    }
}
