use std::collections::HashMap;

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

#[must_use]
pub fn parse_codex_sse_line(line: &str) -> Option<Value> {
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
    pub thought_signature: String,
    pub reasoning: String,
    pub tool_ids: Vec<String>,
    pub history_key: String,
    pub full_text: String,
    pub streamed_tools: Vec<GeminiFunctionCall>,
    pub finish_reason: Option<String>,
    pub codex: CodexAccumulator,
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
            thought_signature: String::new(),
            reasoning: String::new(),
            tool_ids: Vec::new(),
            history_key: String::new(),
            full_text: String::new(),
            streamed_tools: Vec::new(),
            finish_reason: None,
            codex: CodexAccumulator::default(),
        }
    }
}

fn part_signature(part: &Value) -> Option<&str> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .and_then(Value::as_str)
}

fn note_part_signature(
    state_sig: &mut String,
    reasoning: &mut String,
    tool_ids: &[String],
    history_key: &str,
    part: &Value,
) {
    if part.get("thought").and_then(Value::as_bool) == Some(true)
        && let Some(text) = part.get("text").and_then(Value::as_str)
    {
        reasoning.push_str(text);
    }
    if let Some(sig) = part_signature(part)
        && !sig.is_empty()
    {
        *state_sig = sig.to_string();
        for id in tool_ids {
            crate::utils::history_cache::record_tool_signature(id, sig, reasoning);
        }
        crate::utils::history_cache::record_history(history_key, sig, reasoning);
    }
}

fn note_tool_call(state_sig: &str, tool_ids: &mut Vec<String>, tool_id: &str) {
    tool_ids.push(tool_id.to_string());
    crate::utils::history_cache::record_tool_signature(tool_id, state_sig, "");
}

fn gemini_part_to_anthropic(part: &Value, state: &mut AnthropicStreamState) -> Vec<String> {
    let mut events = Vec::new();
    let is_thought = part.get("thought").and_then(Value::as_bool) == Some(true);
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if !is_thought {
            state.full_text.push_str(text);
        }
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

    note_part_signature(
        &mut state.thought_signature,
        &mut state.reasoning,
        &state.tool_ids,
        &state.history_key,
        part,
    );
    if let Some(fc) = part.get("functionCall") {
        let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
        let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
        state.streamed_tools.push(GeminiFunctionCall {
            name: name.to_string(),
            args: args.clone(),
        });
        let tool_id = format!("toolu_{}", uuid::Uuid::new_v4());
        note_tool_call(&state.thought_signature, &mut state.tool_ids, &tool_id);
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": state.block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {}
            }
        });
        events.push(format!("event: content_block_start\ndata: {block_start}\n\n"));
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": state.block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": args.to_string()
            }
        });
        events.push(format!("event: content_block_delta\ndata: {delta}\n\n"));
        let block_stop = serde_json::json!({
            "type": "content_block_stop",
            "index": state.block_index
        });
        events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));
        state.block_index += 1;
    }
    events
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
            events.extend(gemini_part_to_anthropic(part, state));
        }
    }

    let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(Value::as_str);
    if (finish_reason.is_some() || chunk.get("response").is_none()) && !state.finished {
        state.finished = true;
        state.finish_reason = finish_reason.map(str::to_string);
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
    pub thought_signature: String,
    pub reasoning: String,
    pub tool_ids: Vec<String>,
    pub history_key: String,
    pub reasoning_excluded: bool,
    pub full_text: String,
    pub streamed_tools: Vec<GeminiFunctionCall>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub codex: CodexAccumulator,
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
            thought_signature: String::new(),
            reasoning: String::new(),
            tool_ids: Vec::new(),
            history_key: String::new(),
            reasoning_excluded: false,
            full_text: String::new(),
            streamed_tools: Vec::new(),
            finish_reason: None,
            prompt_tokens: 0,
            output_tokens: 0,
            codex: CodexAccumulator::default(),
        }
    }
}

#[must_use]
pub fn reasoning_format_for(model: &str) -> &'static str {
    let lower = model.to_ascii_lowercase();
    if lower.contains("gpt-5") || lower.contains("codex") {
        return "openai-codex-v1";
    }
    if lower.contains("claude")
        || lower.contains("opus")
        || lower.contains("sonnet")
        || lower.contains("fable")
        || lower.contains("haiku")
    {
        return "anthropic-claude-v1";
    }
    "google-gemini-v1"
}

#[must_use]
pub fn reasoning_details_chunk(
    completion_id: &str,
    created: u64,
    model: &str,
    text: &str,
    signature: &str,
) -> Option<serde_json::Value> {
    if signature.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "id": serde_json::Value::Null,
                    "text": text,
                    "format": reasoning_format_for(model),
                    "index": 0,
                    "signature": signature
                }]
            },
            "finish_reason": serde_json::Value::Null
        }]
    }))
}

fn gemini_part_to_openai(part: &Value, state: &mut OpenAiStreamState) -> Vec<String> {
    let mut events = Vec::new();
    note_part_signature(
        &mut state.thought_signature,
        &mut state.reasoning,
        &state.tool_ids,
        &state.history_key,
        part,
    );
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if part.get("thought").and_then(Value::as_bool) != Some(true) {
            state.full_text.push_str(text);
        }
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

    if let Some(fc) = part.get("functionCall") {
        let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
        let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
        state.streamed_tools.push(GeminiFunctionCall {
            name: name.to_string(),
            args: args.clone(),
        });
        let call_id = format!("call_{}", uuid::Uuid::new_v4());
        note_tool_call(&state.thought_signature, &mut state.tool_ids, &call_id);
        let tc_chunk = serde_json::json!({
            "id": &state.completion_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": &state.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args.to_string()
                        }
                    }]
                },
                "finish_reason": serde_json::Value::Null
            }]
        });
        events.push(format!("data: {tc_chunk}\n\n"));
    }
    events
}

#[must_use]
pub fn gemini_to_openai_sse(chunk: &Value, state: &mut OpenAiStreamState) -> Vec<String> {
    let mut events = Vec::new();

    if let Some(usage) = chunk.get("response").and_then(|r| r.get("usageMetadata")) {
        if let Some(prompt) = usage.get("promptTokenCount").and_then(Value::as_u64) {
            state.prompt_tokens = u32::try_from(prompt).unwrap_or(0);
        }
        if let Some(cand) = usage.get("candidatesTokenCount").and_then(Value::as_u64) {
            state.output_tokens = u32::try_from(cand).unwrap_or(0);
        }
    }

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
            events.extend(gemini_part_to_openai(part, state));
        }
    }

    let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(Value::as_str);
    if (finish_reason.is_some() || chunk.get("response").is_none()) && !state.finished {
        state.finished = true;
        state.finish_reason = finish_reason.map(str::to_string);
        finish_openai_gemini_turn(state, &mut events);
    }

    events
}

fn finish_openai_gemini_turn(state: &OpenAiStreamState, events: &mut Vec<String>) {
    if !state.reasoning_excluded {
        let token = crate::utils::history_cache::mint_reasoning_token(&state.thought_signature, &state.reasoning);
        if let Some(sig_chunk) = reasoning_details_chunk(&state.completion_id, state.created, &state.model, "", &token)
        {
            events.push(format!("data: {sig_chunk}\n\n"));
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodexToolAssembly {
    pub id: String,
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, Default)]
pub struct CodexAccumulator {
    pub tools: Vec<CodexToolAssembly>,
    pub index_by_item: HashMap<String, usize>,
    pub reasoning: String,
    pub encrypted: Vec<String>,
    pub usage: Option<Value>,
}

fn note_codex_item_done(acc: &mut CodexAccumulator, item: Option<&Value>) {
    let item_type = item.and_then(|i| i.get("type")).and_then(Value::as_str);
    let str_field = |name: &str| item.and_then(|i| i.get(name)).and_then(Value::as_str);
    if item_type == Some("function_call") {
        let item_id = str_field("id");
        let found = item_id.and_then(|id| acc.index_by_item.get(id)).copied();
        if let Some(index) = found
            && let Some(tool) = acc.tools.get_mut(index)
        {
            if let Some(args) = str_field("arguments") {
                tool.args = args.to_string();
            }
            if let Some(id) = str_field("call_id") {
                tool.id = id.to_string();
            }
            if let Some(name) = str_field("name") {
                tool.name = name.to_string();
            }
        } else {
            let index = acc.tools.len();
            acc.tools.push(CodexToolAssembly {
                id: str_field("call_id").unwrap_or("").to_string(),
                name: str_field("name").unwrap_or("").to_string(),
                args: str_field("arguments").unwrap_or("").to_string(),
            });
            if let Some(item_id) = item_id {
                acc.index_by_item.insert(item_id.to_string(), index);
            }
        }
    } else if item_type == Some("reasoning") {
        if let Some(blob) = str_field("encrypted_content") {
            acc.encrypted.push(blob.to_string());
        }
        if acc.reasoning.is_empty()
            && let Some(summary) = item.and_then(|i| i.get("summary")).and_then(Value::as_array)
        {
            acc.reasoning = summary
                .iter()
                .filter_map(|s| s.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("");
        }
    }
}

/// Observe one Codex Responses frame for later cache populate, mirroring TS `streamCodexResponse`
/// accumulation. Emission is untouched; this only records.
pub fn accumulate_codex_frame(acc: &mut CodexAccumulator, text: &mut String, frame: &Value) {
    match frame.get("type").and_then(Value::as_str).unwrap_or("") {
        "response.output_text.delta" => {
            if let Some(delta) = frame.get("delta").and_then(Value::as_str) {
                text.push_str(delta);
            }
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" | "response.thinking_text.delta" => {
            if let Some(delta) = frame.get("delta").and_then(Value::as_str) {
                acc.reasoning.push_str(delta);
            }
        }
        "response.output_item.added" => {
            let item = frame.get("item");
            if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call") {
                let index = acc.tools.len();
                acc.tools.push(CodexToolAssembly {
                    id: item
                        .and_then(|i| i.get("call_id"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    name: item
                        .and_then(|i| i.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    args: String::new(),
                });
                if let Some(item_id) = item.and_then(|i| i.get("id")).and_then(Value::as_str) {
                    acc.index_by_item.insert(item_id.to_string(), index);
                }
            }
        }
        "response.function_call_arguments.delta" => {
            let delta = frame.get("delta").and_then(Value::as_str).unwrap_or("");
            if let Some(item_id) = frame.get("item_id").and_then(Value::as_str)
                && let Some(&index) = acc.index_by_item.get(item_id)
                && let Some(tool) = acc.tools.get_mut(index)
            {
                tool.args.push_str(delta);
            }
        }
        "response.output_item.done" => note_codex_item_done(acc, frame.get("item")),
        "response.completed" => {
            acc.usage = frame.get("response").and_then(|r| r.get("usage")).cloned();
        }
        _ => {}
    }
}

#[derive(Debug, Clone)]
pub struct CollectedGemini {
    pub text: String,
    pub reasoning: String,
    pub function_calls: Vec<GeminiFunctionCall>,
    pub thought_signature: Option<String>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
}

#[must_use]
pub fn collect_gemini_response(chunks: &[Value]) -> CollectedGemini {
    let mut full_text = String::new();
    let mut reasoning = String::new();
    let mut function_calls = Vec::new();
    let mut thought_signature: Option<String> = None;
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
                        if part.get("thought").and_then(Value::as_bool) == Some(true) {
                            reasoning.push_str(t);
                        } else {
                            full_text.push_str(t);
                        }
                    }
                    if let Some(sig) = part_signature(part)
                        && !sig.is_empty()
                    {
                        thought_signature = Some(sig.to_string());
                    }
                    if let Some(fc) = part.get("functionCall")
                        && let Some(name) = fc.get("name").and_then(Value::as_str)
                    {
                        let args = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
                        function_calls.push(GeminiFunctionCall {
                            name: name.to_string(),
                            args,
                        });
                    }
                }
            }
        }
    }

    CollectedGemini {
        text: full_text,
        reasoning,
        function_calls,
        thought_signature,
        finish_reason,
        prompt_tokens,
        output_tokens,
    }
}

#[must_use]
pub fn codex_to_openai_sse(chunk: &Value, state: &mut OpenAiStreamState) -> Vec<String> {
    let mut events = Vec::new();
    accumulate_codex_frame(&mut state.codex, &mut state.full_text, chunk);
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
    if chunk_type == "response.output_text.delta" {
        if let Some(text) = chunk.get("delta").and_then(Value::as_str) {
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
    } else if chunk_type == "response.function_call_arguments.delta" {
        if let Some(args_delta) = chunk.get("delta").and_then(Value::as_str) {
            let tc_chunk = serde_json::json!({
                "id": &state.completion_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": &state.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": args_delta
                            }
                        }]
                    },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            events.push(format!("data: {tc_chunk}\n\n"));
        }
    } else if chunk_type == "response.completed" && !state.finished {
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
    events
}

#[must_use]
pub fn codex_to_anthropic_sse(chunk: &Value, state: &mut AnthropicStreamState) -> Vec<String> {
    let mut events = Vec::new();

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

    accumulate_codex_frame(&mut state.codex, &mut state.full_text, chunk);
    let chunk_type = chunk.get("type").and_then(Value::as_str).unwrap_or("");
    if chunk_type == "response.output_text.delta" {
        if !state.text_block_started {
            state.text_block_started = true;
            let block_start = serde_json::json!({
                "type": "content_block_start",
                "index": state.block_index,
                "content_block": { "type": "text", "text": "" }
            });
            events.push(format!("event: content_block_start\ndata: {block_start}\n\n"));
        }
        if let Some(text) = chunk.get("delta").and_then(Value::as_str) {
            let delta = serde_json::json!({
                "type": "content_block_delta",
                "index": state.block_index,
                "delta": { "type": "text_delta", "text": text }
            });
            events.push(format!("event: content_block_delta\ndata: {delta}\n\n"));
        }
    } else if chunk_type == "response.output_item.added" {
        let is_func = chunk.get("item").and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call");
        if is_func {
            let item = &chunk["item"];
            let name = item.get("name").and_then(Value::as_str).unwrap_or("");
            let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
            let block_start = serde_json::json!({
                "type": "content_block_start",
                "index": state.block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {}
                }
            });
            events.push(format!("event: content_block_start\ndata: {block_start}\n\n"));
        }
    } else if chunk_type == "response.function_call_arguments.delta" {
        if let Some(args_delta) = chunk.get("delta").and_then(Value::as_str) {
            let delta = serde_json::json!({
                "type": "content_block_delta",
                "index": state.block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": args_delta
                }
            });
            events.push(format!("event: content_block_delta\ndata: {delta}\n\n"));
        }
    } else if chunk_type == "response.output_item.done" {
        let is_func = chunk.get("item").and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call");
        if is_func {
            let block_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": state.block_index
            });
            events.push(format!("event: content_block_stop\ndata: {block_stop}\n\n"));
            state.block_index += 1;
        }
    } else if chunk_type == "response.completed" && !state.finished {
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
            "delta": { "stop_reason": "end_turn", "stop_sequence": serde_json::Value::Null },
            "usage": { "output_tokens": state.output_tokens }
        });
        events.push(format!("event: message_delta\ndata: {msg_delta}\n\n"));
        events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());
    }

    events
}

#[must_use]
pub fn collect_codex_response(chunks: &[Value]) -> (String, Option<String>, u32, u32) {
    let mut full_text = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_type = chunk.get("type").and_then(Value::as_str).unwrap_or("");
        if chunk_type == "response.output_text.delta" {
            if let Some(t) = chunk.get("delta").and_then(Value::as_str) {
                full_text.push_str(t);
            }
        } else if chunk_type == "response.completed" {
            let usage_opt = chunk.get("response").and_then(|r| r.get("usage"));
            if let Some(usage) = usage_opt {
                input_tokens = usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .map_or(0, |v| u32::try_from(v).unwrap_or(0));
                output_tokens = usage
                    .get("output_tokens")
                    .and_then(Value::as_u64)
                    .map_or(0, |v| u32::try_from(v).unwrap_or(0));
            }
        }
    }
    (full_text, Some("stop".to_string()), input_tokens, output_tokens)
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
    fn test_gemini_openai_translator_accumulates_turn() {
        let mut state = OpenAiStreamState::new("test-cmpl-acc", "m");
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [
                        { "text": "hello" },
                        { "text": "secret", "thought": true, "thoughtSignature": "sig_acc_1" },
                        { "functionCall": { "name": "read", "args": { "p": "v" } } }
                    ] },
                    "finishReason": "STOP"
                }],
                "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 3 }
            }
        });
        let _ = gemini_to_openai_sse(&chunk, &mut state);
        assert_eq!(state.full_text, "hello");
        assert_eq!(state.reasoning, "secret");
        assert_eq!(state.thought_signature, "sig_acc_1");
        assert_eq!(state.streamed_tools.len(), 1);
        assert_eq!(state.streamed_tools[0].name, "read");
        assert_eq!(state.streamed_tools[0].args, serde_json::json!({ "p": "v" }));
        assert_eq!(state.prompt_tokens, 5);
        assert_eq!(state.output_tokens, 3);
        assert_eq!(state.finish_reason.as_deref(), Some("STOP"));
        assert!(state.finished);
    }

    #[test]
    fn test_codex_frame_accumulation_mirrors_ts() {
        let mut acc = CodexAccumulator::default();
        let mut text = String::new();
        let frames = [
            serde_json::json!({ "type": "response.output_text.delta", "delta": "hel" }),
            serde_json::json!({ "type": "response.output_text.delta", "delta": "lo" }),
            serde_json::json!({ "type": "response.reasoning_summary_text.delta", "delta": "pla" }),
            serde_json::json!({
                "type": "response.output_item.added",
                "item": { "type": "function_call", "id": "item1", "call_id": "call_1", "name": "run" }
            }),
            serde_json::json!({ "type": "response.function_call_arguments.delta", "item_id": "item1", "delta": "{\"c\":" }),
            serde_json::json!({ "type": "response.function_call_arguments.delta", "item_id": "item1", "delta": "\"ls\"}" }),
            serde_json::json!({
                "type": "response.output_item.done",
                "item": { "type": "function_call", "id": "item1", "call_id": "call_1", "name": "run", "arguments": "{\"c\":\"ls\"}" }
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "item": { "type": "reasoning", "encrypted_content": "enc1", "summary": [{ "text": "full plan" }] }
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": { "usage": { "input_tokens": 30, "output_tokens": 12 } }
            }),
        ];
        for frame in &frames {
            accumulate_codex_frame(&mut acc, &mut text, frame);
        }
        assert_eq!(text, "hello");
        assert_eq!(acc.reasoning, "pla");
        assert_eq!(acc.tools.len(), 1);
        assert_eq!(acc.tools[0].id, "call_1");
        assert_eq!(acc.tools[0].name, "run");
        assert_eq!(acc.tools[0].args, "{\"c\":\"ls\"}");
        assert_eq!(acc.encrypted, vec!["enc1".to_string()]);
        assert_eq!(
            acc.usage
                .as_ref()
                .and_then(|u| u.get("input_tokens"))
                .and_then(Value::as_u64),
            Some(30)
        );
    }

    #[test]
    fn test_codex_done_without_added_still_collects_tool() {
        let mut acc = CodexAccumulator::default();
        let mut text = String::new();
        accumulate_codex_frame(
            &mut acc,
            &mut text,
            &serde_json::json!({
                "type": "response.output_item.done",
                "item": { "type": "function_call", "call_id": "call_9", "name": "late", "arguments": "{}" }
            }),
        );
        assert_eq!(acc.tools.len(), 1);
        assert_eq!(acc.tools[0].id, "call_9");
    }

    #[test]
    fn test_codex_translators_feed_the_accumulator() {
        let mut openai = OpenAiStreamState::new("c", "m");
        let _ = codex_to_openai_sse(
            &serde_json::json!({ "type": "response.output_text.delta", "delta": "abc" }),
            &mut openai,
        );
        assert_eq!(openai.full_text, "abc");
        let mut anthropic = AnthropicStreamState::new("msg", "m");
        let _ = codex_to_anthropic_sse(
            &serde_json::json!({ "type": "response.output_text.delta", "delta": "xyz" }),
            &mut anthropic,
        );
        assert_eq!(anthropic.full_text, "xyz");
    }

    #[test]
    fn test_gemini_anthropic_translator_accumulates_turn() {
        let mut state = AnthropicStreamState::new("msg-acc", "m");
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [
                        { "text": "visible" },
                        { "text": "hidden", "thought": true, "thoughtSignature": "sig_acc_2" },
                        { "functionCall": { "name": "write", "args": {} } }
                    ] },
                    "finishReason": "MAX_TOKENS"
                }],
                "usageMetadata": { "promptTokenCount": 8, "candidatesTokenCount": 4 }
            }
        });
        let _ = gemini_to_anthropic_sse(&chunk, &mut state);
        assert_eq!(state.full_text, "visible");
        assert_eq!(state.reasoning, "hidden");
        assert_eq!(state.streamed_tools.len(), 1);
        assert_eq!(state.streamed_tools[0].name, "write");
        assert_eq!(state.input_tokens, 8);
        assert_eq!(state.output_tokens, 4);
        assert_eq!(state.finish_reason.as_deref(), Some("MAX_TOKENS"));
        assert!(state.finished);
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

        let res = collect_gemini_response(&[chunk1, chunk2]);
        assert_eq!(res.text, "foo bar");
        assert_eq!(res.finish_reason.as_deref(), Some("STOP"));
        assert_eq!(res.prompt_tokens, 5);
        assert_eq!(res.output_tokens, 2);
    }

    #[test]
    fn test_gemini_stream_records_signature_for_emitted_tool_id() {
        let mut state = OpenAiStreamState::new("test-cmpl-sig", "gemini-3.8-flash-high");
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [{
                        "functionCall": { "name": "read", "args": {}, "id": "upstream-1" },
                        "thoughtSignature": "pxsig_stream_1"
                    }] }
                }]
            }
        });
        let events = gemini_to_openai_sse(&chunk, &mut state);
        let mut tool_id = String::new();
        for ev in &events {
            if let Some(json_part) = ev.strip_prefix("data: ") {
                let val: serde_json::Value = serde_json::from_str(json_part.trim()).unwrap();
                if let Some(id) = val
                    .pointer("/choices/0/delta/tool_calls/0/id")
                    .and_then(serde_json::Value::as_str)
                {
                    tool_id = id.to_string();
                }
            }
        }
        assert_ne!(tool_id, "");
        let entry = crate::utils::history_cache::lookup_tool_signature(&tool_id).unwrap();
        assert_eq!(entry.thought_signature, "pxsig_stream_1");
    }

    #[test]
    fn test_collect_gemini_response_captures_signature() {
        let chunk = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": { "parts": [{
                        "functionCall": { "name": "read", "args": {} },
                        "thoughtSignature": "pxsig_buffered_1"
                    }] }
                }]
            }
        });
        let res = collect_gemini_response(&[chunk]);
        assert_eq!(res.thought_signature.as_deref(), Some("pxsig_buffered_1"));
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

    #[test]
    fn test_codex_to_openai_sse_events() {
        let mut state = OpenAiStreamState::new("test-cmpl-codex", "gpt-5.5");
        let delta = serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "codex output"
        });
        let done = serde_json::json!({
            "type": "response.completed"
        });

        let events1 = codex_to_openai_sse(&delta, &mut state);
        let events2 = codex_to_openai_sse(&done, &mut state);

        let all = [events1, events2].concat().join("\n");
        assert!(all.contains("codex output"));
        assert!(all.contains("[DONE]"));
    }

    #[test]
    fn test_codex_to_anthropic_sse_events() {
        let mut state = AnthropicStreamState::new("test-msg-codex", "gpt-5.5");
        let delta = serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "codex anthropic text"
        });
        let done = serde_json::json!({
            "type": "response.completed"
        });

        let events1 = codex_to_anthropic_sse(&delta, &mut state);
        let events2 = codex_to_anthropic_sse(&done, &mut state);

        let all = [events1, events2].concat().join("\n");
        assert!(all.contains("message_start"));
        assert!(all.contains("codex anthropic text"));
        assert!(all.contains("message_stop"));
    }
}
