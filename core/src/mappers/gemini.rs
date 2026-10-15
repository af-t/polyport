use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mappers::media::{MediaError, parse_anthropic_media_block, parse_openai_media_block, render_for_gemini};
use crate::utils::history_cache::{
    DUMMY_SIGNATURE, FALLBACK_SIGNATURE, HistoryContext, history_key_chain, lookup_history, lookup_tool_signature,
    resolve_reasoning_token,
};

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

#[must_use]
pub fn find_tool_name_by_id(messages: &Value, tool_use_id: &str) -> String {
    let Some(arr) = messages.as_array() else {
        return String::new();
    };
    for m in arr {
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            for tc in calls {
                if tc.get("id").and_then(Value::as_str) == Some(tool_use_id) {
                    return tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                }
            }
        }
        if let Some(blocks) = m.get("content").and_then(Value::as_array) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_use")
                    && block.get("id").and_then(Value::as_str) == Some(tool_use_id)
                {
                    return block.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                }
            }
        }
        if m.get("role").and_then(Value::as_str) == Some("tool")
            && m.get("tool_call_id").and_then(Value::as_str) == Some(tool_use_id)
        {
            return m.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        }
    }
    String::new()
}

#[must_use]
pub fn resolve_function_response_name(messages: &Value, tool_use_id: &str, fallback: Option<&str>) -> String {
    let found = find_tool_name_by_id(messages, tool_use_id);
    if !found.is_empty() {
        return found;
    }
    if let Some(f) = fallback.filter(|f| !f.is_empty()) {
        return f.to_string();
    }
    "unknown_tool".to_string()
}

/// Maps content (string or array of blocks) into Gemini parts.
///
/// # Errors
/// Returns `MapperError` if any media item cannot be rendered.
pub fn map_content_parts(
    content: &Value,
    messages: &Value,
    injected_sig: Option<&str>,
    is_thinking_requested: bool,
) -> Result<Vec<Value>, MapperError> {
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
                    let mut sig: Option<String> = injected_sig.map(str::to_string);
                    if sig.is_none() {
                        let raw = block
                            .get("signature")
                            .or_else(|| block.get("thoughtSignature"))
                            .or_else(|| block.get("thought_signature"))
                            .and_then(Value::as_str);
                        if let Some(raw_str) = raw {
                            sig = resolve_reasoning_token(raw_str).map(|r| r.signature);
                        }
                    }

                    if let Some(sig_str) = sig.as_deref()
                        && sig_str != DUMMY_SIGNATURE
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
                let tool_use_id = block.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                let name =
                    resolve_function_response_name(messages, tool_use_id, block.get("name").and_then(Value::as_str));
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

fn signature_for_message(m: &Value, content_val: &Value) -> Option<String> {
    if let Some(tool_calls) = m.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            if let Some(id) = tc.get("id").and_then(Value::as_str)
                && let Some(entry) = lookup_tool_signature(id)
            {
                return Some(entry.thought_signature);
            }
        }
    }
    if let Some(blocks) = content_val.as_array() {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use")
                && let Some(id) = block.get("id").and_then(Value::as_str)
                && let Some(entry) = lookup_tool_signature(id)
            {
                return Some(entry.thought_signature);
            }
        }
    }
    None
}

fn client_reasoning_token(m: &Value) -> (Option<String>, Option<String>) {
    let mut token: Option<String> = None;
    if let Some(blocks) = m.get("content").and_then(Value::as_array) {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("thinking") && token.is_none() {
                token = block
                    .get("signature")
                    .or_else(|| block.get("thoughtSignature"))
                    .or_else(|| block.get("thought_signature"))
                    .and_then(Value::as_str)
                    .filter(|s| *s != DUMMY_SIGNATURE)
                    .map(str::to_string);
            }
        }
    }
    let mut details_text: Option<String> = None;
    let details = m.get("reasoning_details");
    if let Some(arr) = details.and_then(Value::as_array) {
        for d in arr {
            if token.is_none() {
                token = d
                    .get("signature")
                    .and_then(Value::as_str)
                    .filter(|s| *s != DUMMY_SIGNATURE)
                    .map(str::to_string);
            }
            if details_text.is_none() {
                details_text = d
                    .get("reasoning_content")
                    .or_else(|| d.get("text"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
        }
    } else if let Some(single) = details.filter(|d| d.is_object()) {
        if token.is_none() {
            token = single
                .get("signature")
                .and_then(Value::as_str)
                .filter(|s| *s != DUMMY_SIGNATURE)
                .map(str::to_string);
        }
        details_text = single
            .get("reasoning_content")
            .or_else(|| single.get("text"))
            .and_then(Value::as_str)
            .map(str::to_string);
    }
    let text = m
        .get("reasoning")
        .or_else(|| m.get("reasoning_content"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(details_text);
    (token, text)
}

fn inbound_signature(
    m: &Value,
    content_val: &Value,
    is_assistant: bool,
    is_thinking_requested: bool,
) -> (Option<String>, Option<String>) {
    let mut signature: Option<String> = None;
    let mut reasoning: Option<String> = None;
    if is_assistant {
        let (token, text) = client_reasoning_token(m);
        reasoning = text;
        if is_thinking_requested
            && let Some(tok) = token
            && let Some(resolved) = resolve_reasoning_token(&tok)
        {
            signature = Some(resolved.signature);
            if reasoning.as_deref().is_none_or(str::is_empty) {
                reasoning = Some(resolved.reasoning);
            }
        }
    }
    if signature.is_none() {
        signature = signature_for_message(m, content_val);
    }
    (signature, reasoning)
}

fn push_system_text(system_texts: &mut Vec<String>, content_val: &Value) {
    if let Some(txt) = content_val.as_str() {
        system_texts.push(txt.to_string());
    } else if let Some(sub_arr) = content_val.as_array() {
        for block in sub_arr {
            if let Some(txt) = block.get("text").and_then(Value::as_str) {
                system_texts.push(txt.to_string());
            }
        }
    }
}

fn map_tool_message(m: &Value, messages: &Value, content_val: &Value) -> GeminiContent {
    let tool_call_id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
    let name = resolve_function_response_name(messages, tool_call_id, m.get("name").and_then(Value::as_str));
    let resp = parse_function_response(content_val);
    GeminiContent {
        role: "user".to_string(),
        parts: vec![serde_json::json!({
            "functionResponse": {
                "name": name,
                "response": resp,
                "id": tool_call_id
            }
        })],
    }
}

fn append_tool_calls(parts: &mut Vec<Value>, m: &Value) {
    let Some(tool_calls) = m.get("tool_calls").and_then(Value::as_array) else {
        return;
    };
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

fn prefix_history_entry(
    idx: usize,
    key_chain: Option<&Vec<String>>,
) -> Option<crate::utils::history_cache::HistoryEntry> {
    key_chain?.get(idx).and_then(|k| lookup_history(k))
}

fn attach_thought_signature(
    parts: &mut Vec<Value>,
    signature: Option<String>,
    is_model: bool,
    is_thinking_requested: bool,
) {
    let has_function_call = parts.iter().any(|p| p.get("functionCall").is_some());
    let mut final_signature = signature;
    if final_signature.as_deref().is_none_or(|s| s == DUMMY_SIGNATURE) {
        if has_function_call && is_model && is_thinking_requested {
            final_signature = Some(FALLBACK_SIGNATURE.to_string());
        } else {
            final_signature = None;
        }
    }
    if let Some(sig) = final_signature
        && !parts.is_empty()
    {
        let mut injected = false;
        for part in &mut *parts {
            if part.get("functionCall").is_some() {
                part["thoughtSignature"] = Value::String(sig.clone());
                part["thought_signature"] = Value::String(sig.clone());
                injected = true;
            }
        }
        if !injected && let Some(last) = parts.last_mut() {
            last["thoughtSignature"] = Value::String(sig.clone());
            last["thought_signature"] = Value::String(sig);
        }
    }
}

/// Maps client messages to Gemini contents and system instruction.
///
/// # Errors
/// Returns `MapperError` on media translation failure.
pub fn map_messages_to_gemini(
    messages: &Value,
    system_prompt: Option<&str>,
    is_thinking_requested: bool,
    history: Option<&HistoryContext>,
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

    let key_chain: Option<Vec<String>> = history.map(|h| history_key_chain(&h.model, messages, &h.system, &h.tools));

    let mut contents = Vec::new();

    for (idx, m) in arr.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content_val = m.get("content").unwrap_or(&Value::Null);

        if role == "system" {
            push_system_text(&mut system_texts, content_val);
            continue;
        }

        if role == "tool" {
            contents.push(map_tool_message(m, messages, content_val));
            continue;
        }

        let is_assistant = role == "assistant";
        let target_role = if is_assistant { "model" } else { "user" };

        let (mut signature_to_inject, reasoning_text) =
            inbound_signature(m, content_val, is_assistant, is_thinking_requested);

        let mut parts = Vec::new();
        if signature_to_inject.is_none()
            && target_role == "model"
            && let Some(entry) = prefix_history_entry(idx, key_chain.as_ref())
        {
            let has_thinking = content_val.as_array().is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
            });
            if !entry.reasoning.is_empty() && !has_thinking {
                parts.push(serde_json::json!({
                    "text": entry.reasoning,
                    "thought": true,
                    "thoughtSignature": entry.thought_signature,
                    "thought_signature": entry.thought_signature
                }));
            }
            signature_to_inject = Some(entry.thought_signature);
        }

        parts.extend(map_content_parts(
            content_val,
            messages,
            signature_to_inject.as_deref(),
            is_thinking_requested,
        )?);

        if is_assistant {
            append_tool_calls(&mut parts, m);
        }

        let has_thought_part = parts
            .iter()
            .any(|p| p.get("thought").and_then(Value::as_bool) == Some(true));
        if target_role == "model"
            && is_thinking_requested
            && !has_thought_part
            && let Some(text) = reasoning_text.as_deref().filter(|s| !s.is_empty())
            && let Some(sig) = signature_to_inject.clone().filter(|s| s != DUMMY_SIGNATURE)
        {
            parts.insert(
                0,
                serde_json::json!({
                    "text": text,
                    "thought": true,
                    "thoughtSignature": sig,
                    "thought_signature": sig
                }),
            );
        }

        attach_thought_signature(
            &mut parts,
            signature_to_inject,
            target_role == "model",
            is_thinking_requested,
        );

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

fn decode_pointer_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn resolve_pointer<'a>(root: &'a Value, pointer: &str) -> Option<&'a Value> {
    if !pointer.starts_with("#/") {
        return None;
    }
    let mut node = root;
    for raw in pointer[2..].split('/') {
        let key = decode_pointer_token(raw);
        node = node.get(&key)?;
    }
    Some(node)
}

fn inline_refs_walk(node: &Value, root: &Value, seen: &mut Vec<String>) -> Value {
    if let Some(arr) = node.as_array() {
        return Value::Array(arr.iter().map(|n| inline_refs_walk(n, root, seen)).collect());
    }
    let Some(obj) = node.as_object() else {
        return node.clone();
    };
    if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
        if seen.iter().any(|s| s == ref_str) {
            return Value::Object(serde_json::Map::new());
        }
        let mut siblings = obj.clone();
        siblings.remove("$ref");
        let siblings_val = Value::Object(siblings);
        let Some(target) = resolve_pointer(root, ref_str) else {
            return inline_refs_walk(&siblings_val, root, seen);
        };
        let mut merged = serde_json::Map::new();
        if let Some(target_obj) = target.as_object() {
            merged.extend(target_obj.clone());
        }
        if let Some(siblings_obj) = siblings_val.as_object() {
            merged.extend(siblings_obj.clone());
        }
        seen.push(ref_str.to_string());
        let out = inline_refs_walk(&Value::Object(merged), root, seen);
        seen.pop();
        return out;
    }
    let mut out = serde_json::Map::new();
    for (k, v) in obj {
        if k == "$defs" || k == "definitions" {
            continue;
        }
        out.insert(k.clone(), inline_refs_walk(v, root, seen));
    }
    Value::Object(out)
}

/// Inlines same-document `$ref`s, drops `$defs`/`definitions`.
#[must_use]
pub fn inline_schema_refs(root: &Value) -> Value {
    inline_refs_walk(root, root, &mut Vec::new())
}

fn collapse_enum_union(branches: &[Value]) -> Option<(String, Vec<Value>)> {
    if branches.is_empty() {
        return None;
    }
    let allowed = ["type", "enum", "const", "description", "title", "default"];
    let mut branch_type: Option<String> = None;
    let mut values: Vec<Value> = Vec::new();
    for b in branches {
        let obj = b.as_object()?;
        let t = obj.get("type")?.as_str()?;
        if let Some(existing) = &branch_type {
            if existing != t {
                return None;
            }
        } else {
            branch_type = Some(t.to_string());
        }
        if obj.keys().any(|k| !allowed.contains(&k.as_str())) {
            return None;
        }
        if let Some(enums) = obj.get("enum").and_then(Value::as_array) {
            values.extend(enums.iter().cloned());
        } else {
            let c = obj.get("const")?;
            values.push(c.clone());
        }
    }
    let branch_type = branch_type?;
    if values.is_empty() {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for v in values {
        let key = serde_json::to_string(&v).unwrap_or_default();
        if seen.insert(key) {
            deduped.push(v);
        }
    }
    Some((branch_type, deduped))
}

fn normalize_schema_inner(schema: &Value) -> Value {
    if let Some(arr) = schema.as_array() {
        return Value::Array(arr.iter().map(normalize_schema_inner).collect());
    }
    let Some(obj) = schema.as_object() else {
        return schema.clone();
    };
    let mut working = obj.clone();

    if let Some(c) = working.remove("const") {
        working.insert("enum".to_string(), Value::Array(vec![c]));
    }

    if let Some(types) = working.get("type").and_then(Value::as_array).cloned() {
        let lowered: Vec<Value> = types
            .iter()
            .map(|t| {
                t.as_str()
                    .map_or_else(|| t.clone(), |s| Value::String(s.to_ascii_lowercase()))
            })
            .collect();
        if lowered.iter().any(|t| t.as_str() == Some("null")) {
            working.insert("nullable".to_string(), Value::Bool(true));
        }
        let first = lowered.into_iter().find(|t| t.as_str() != Some("null"));
        match first {
            Some(t) => {
                working.insert("type".to_string(), t);
            }
            None => {
                working.remove("type");
            }
        }
    }

    if working.contains_key("any_of")
        && !working.contains_key("anyOf")
        && let Some(v) = working.remove("any_of")
    {
        working.insert("anyOf".to_string(), v);
    }
    if working.contains_key("one_of")
        && !working.contains_key("oneOf")
        && let Some(v) = working.remove("one_of")
    {
        working.insert("oneOf".to_string(), v);
    }
    if working.contains_key("oneOf")
        && !working.contains_key("anyOf")
        && let Some(v) = working.remove("oneOf")
    {
        working.insert("anyOf".to_string(), v);
    }

    let allowed = [
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "properties",
        "required",
        "items",
        "anyOf",
        "default",
    ];
    let mut normalized = serde_json::Map::new();
    for (k, v) in &working {
        if !allowed.contains(&k.as_str()) {
            continue;
        }
        let out = if k == "type" {
            v.as_str()
                .map_or_else(|| v.clone(), |s| Value::String(s.to_ascii_uppercase()))
        } else if k == "properties" {
            match v.as_object() {
                Some(props) => {
                    let mut norm_props = serde_json::Map::new();
                    for (name, val) in props {
                        norm_props.insert(name.clone(), normalize_schema_inner(val));
                    }
                    Value::Object(norm_props)
                }
                None => v.clone(),
            }
        } else if k == "items" {
            normalize_schema_inner(v)
        } else if k == "anyOf" {
            match v.as_array() {
                Some(branches) => Value::Array(branches.iter().map(normalize_schema_inner).collect()),
                None => v.clone(),
            }
        } else {
            v.clone()
        };
        normalized.insert(k.clone(), out);
    }

    if let Some(branches) = normalized.get("anyOf").and_then(Value::as_array).cloned()
        && let Some((union_type, union_enum)) = collapse_enum_union(&branches)
    {
        normalized.remove("anyOf");
        normalized.insert("type".to_string(), Value::String(union_type));
        normalized.insert("enum".to_string(), Value::Array(union_enum));
    }

    Value::Object(normalized)
}

/// Normalizes a tool schema to Gemini's Schema subset.
#[must_use]
pub fn normalize_schema_types(schema: &Value) -> Value {
    normalize_schema_inner(&inline_schema_refs(schema))
}

#[must_use]
pub fn map_openai_tools_for_gemini(tools: &[Value]) -> Option<Vec<Value>> {
    let mut declarations = Vec::new();
    for t in tools {
        if t.get("type").and_then(Value::as_str) == Some("function")
            && let Some(f) = t.get("function")
        {
            let name = f.get("name").and_then(Value::as_str).unwrap_or("");
            let description = f.get("description").and_then(Value::as_str).unwrap_or("");
            let mut decl = serde_json::json!({
                "name": name,
                "description": description,
            });
            if let Some(params) = f.get("parameters") {
                decl["parameters"] = normalize_schema_types(params);
            }
            declarations.push(decl);
        }
    }
    if declarations.is_empty() {
        None
    } else {
        Some(vec![serde_json::json!({ "functionDeclarations": declarations })])
    }
}

#[must_use]
pub fn map_anthropic_tools_for_gemini(tools: &[Value]) -> Option<Vec<Value>> {
    let mut declarations = Vec::new();
    for t in tools {
        let name = t.get("name").and_then(Value::as_str).unwrap_or("");
        let description = t.get("description").and_then(Value::as_str).unwrap_or("");
        let mut decl = serde_json::json!({
            "name": name,
            "description": description,
        });
        if let Some(params) = t.get("input_schema") {
            decl["parameters"] = normalize_schema_types(params);
        }
        declarations.push(decl);
    }
    if declarations.is_empty() {
        None
    } else {
        Some(vec![serde_json::json!({ "functionDeclarations": declarations })])
    }
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
    fn test_tool_role_resolves_name_from_history() {
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ]
            },
            { "role": "tool", "tool_call_id": "call_1", "content": "file text" }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, false, None).unwrap();
        assert_eq!(
            contents[1].parts[0]["functionResponse"]["name"],
            serde_json::json!("read")
        );
    }

    #[test]
    fn test_tool_role_unknown_id_uses_placeholder() {
        let messages = serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "tool", "tool_call_id": "call_missing", "content": "out" }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, false, None).unwrap();
        assert_eq!(
            contents[1].parts[0]["functionResponse"]["name"],
            serde_json::json!("unknown_tool")
        );
    }

    #[test]
    fn test_tool_result_block_resolves_name_from_history() {
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": [{ "type": "tool_use", "id": "call_9", "name": "write", "input": {} }]
            },
            {
                "role": "user",
                "content": [{ "type": "tool_result", "tool_use_id": "call_9", "content": "ok" }]
            }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, false, None).unwrap();
        assert_eq!(
            contents[1].parts[0]["functionResponse"]["name"],
            serde_json::json!("write")
        );
    }

    #[test]
    fn test_function_call_replays_recorded_signature() {
        crate::utils::history_cache::record_tool_signature("test-sig-call-1", "pxsig_turn1", "");
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    { "id": "test-sig-call-1", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ]
            },
            { "role": "tool", "tool_call_id": "test-sig-call-1", "content": "file text" }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, true, None).unwrap();
        let fc = contents[0]
            .parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .unwrap();
        assert_eq!(fc.get("thoughtSignature").and_then(Value::as_str), Some("pxsig_turn1"));
        assert_eq!(fc.get("thought_signature").and_then(Value::as_str), Some("pxsig_turn1"));
    }

    #[test]
    fn test_function_call_without_signature_gets_sentinel_when_thinking() {
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    { "id": "test-sig-call-2", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ]
            }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, true, None).unwrap();
        let fc = contents[0]
            .parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .unwrap();
        assert_eq!(
            fc.get("thoughtSignature").and_then(Value::as_str),
            Some(crate::utils::history_cache::FALLBACK_SIGNATURE)
        );
    }

    #[test]
    fn test_function_call_without_thinking_has_no_signature() {
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    { "id": "test-sig-call-3", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ]
            }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, false, None).unwrap();
        let fc = contents[0]
            .parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .unwrap();
        assert_eq!(fc.get("thoughtSignature"), None);
    }

    #[test]
    fn test_reasoning_details_token_replays_thought() {
        let token = crate::utils::history_cache::mint_reasoning_token("real_sig_echo", "prior reasoning");
        let messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "answer text",
                "reasoning_details": [{ "type": "reasoning.text", "text": "prior reasoning", "signature": token }]
            }
        ]);
        let (_, contents) = map_messages_to_gemini(&messages, None, true, None).unwrap();
        let thought = contents[0].parts.iter().find(|p| p.get("thought").is_some()).unwrap();
        assert_eq!(thought.get("text").and_then(Value::as_str), Some("prior reasoning"));
        assert_eq!(
            thought.get("thoughtSignature").and_then(Value::as_str),
            Some("real_sig_echo")
        );
    }

    #[test]
    fn test_prefix_history_key_replays_thought() {
        use crate::utils::history_cache::{HistoryContext, history_key, record_history};
        let messages = serde_json::json!([
            { "role": "user", "content": "do it" },
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    { "id": "test-prefix-1", "type": "function", "function": { "name": "read", "arguments": "{}" } }
                ]
            }
        ]);
        let ctx = HistoryContext {
            model: "m".to_string(),
            system: serde_json::Value::Null,
            tools: serde_json::Value::Null,
        };
        let prefix = serde_json::json!([{ "role": "user", "content": "do it" }]);
        let key = history_key("m", &prefix, &serde_json::Value::Null, &serde_json::Value::Null);
        record_history(&key, "pxsig_prefix", "prefix thought");
        let (_, contents) = map_messages_to_gemini(&messages, None, true, Some(&ctx)).unwrap();
        let assistant = &contents[1];
        let thought = assistant.parts.iter().find(|p| p.get("thought").is_some()).unwrap();
        assert_eq!(thought.get("text").and_then(Value::as_str), Some("prefix thought"));
        let fc = assistant
            .parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .unwrap();
        assert_eq!(fc.get("thoughtSignature").and_then(Value::as_str), Some("pxsig_prefix"));
    }

    #[test]
    fn test_map_content_parts_string_and_blocks() {
        let string_content = serde_json::json!("hello world");
        let parts = map_content_parts(&string_content, &serde_json::Value::Null, None, false).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].get("text").and_then(serde_json::Value::as_str),
            Some("hello world")
        );

        let blocks_content = serde_json::json!([
            { "type": "text", "text": "foo" },
            { "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } }
        ]);
        let parts2 = map_content_parts(&blocks_content, &serde_json::Value::Null, None, false).unwrap();
        assert_eq!(parts2.len(), 2);
        assert_eq!(parts2[0].get("text").and_then(serde_json::Value::as_str), Some("foo"));
        assert!(parts2[1].get("functionCall").is_some());
    }

    #[test]
    fn test_map_content_parts_thinking_block_resolves_token() {
        let token = crate::utils::history_cache::mint_reasoning_token("real_upstream_sig", "thought process");
        let blocks = serde_json::json!([
            {
                "type": "thinking",
                "thinking": "thought process",
                "signature": token
            }
        ]);
        let parts = map_content_parts(&blocks, &serde_json::Value::Null, None, true).unwrap();
        assert_eq!(parts.len(), 1);
        let part = &parts[0];
        assert_eq!(
            part.get("text").and_then(serde_json::Value::as_str),
            Some("thought process")
        );
        assert_eq!(part.get("thought").and_then(serde_json::Value::as_bool), Some(true));
        assert_eq!(
            part.get("thoughtSignature").and_then(serde_json::Value::as_str),
            Some("real_upstream_sig")
        );
    }

    #[test]
    fn test_map_content_parts_thinking_block_drops_unknown_signature() {
        let blocks = serde_json::json!([
            {
                "type": "thinking",
                "thinking": "thought process",
                "signature": "sig_123"
            }
        ]);
        let parts = map_content_parts(&blocks, &serde_json::Value::Null, None, true).unwrap();
        assert_eq!(parts, [] as [serde_json::Value; 0]);
    }

    #[test]
    fn test_normalize_schema_types_strips_unsupported_keys() {
        let schema = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "count": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "how many"
                }
            },
            "additionalProperties": false
        });
        let out = normalize_schema_types(&schema);
        assert_eq!(out.get("$schema"), None);
        assert_eq!(out.get("additionalProperties"), None);
        assert_eq!(out.get("type").and_then(Value::as_str), Some("OBJECT"));
        let count = &out["properties"]["count"];
        assert_eq!(count.get("exclusiveMinimum"), None);
        assert_eq!(count.get("type").and_then(Value::as_str), Some("NUMBER"));
        assert_eq!(count.get("description").and_then(Value::as_str), Some("how many"));
    }

    #[test]
    fn test_normalize_schema_types_handles_unions_const_and_refs() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "maybe": { "type": ["string", "null"] },
                "fixed": { "const": "yes" },
                "choice": { "oneOf": [{ "type": "string", "enum": ["a"] }] },
                "again": { "$ref": "#/$defs/Name" }
            },
            "$defs": { "Name": { "type": "string" } }
        });
        let out = normalize_schema_types(&schema);
        let props = &out["properties"];
        assert_eq!(props["maybe"]["type"].as_str(), Some("STRING"));
        assert_eq!(props["maybe"]["nullable"].as_bool(), Some(true));
        assert_eq!(props["fixed"]["enum"], serde_json::json!(["yes"]));
        assert_eq!(props["fixed"].get("const"), None);
        assert_eq!(props["choice"]["enum"], serde_json::json!(["a"]));
        assert_eq!(props["choice"].get("oneOf"), None);
        assert_eq!(props["choice"].get("anyOf"), None);
        assert_eq!(props["again"]["type"].as_str(), Some("STRING"));
        assert_eq!(out.get("$defs"), None);
    }

    #[test]
    fn test_openai_tools_normalize_parameters() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read",
                "description": "read a file",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": { "path": { "type": "string" } }
                }
            }
        })];
        let mapped = map_openai_tools_for_gemini(&tools).unwrap();
        let params = &mapped[0]["functionDeclarations"][0]["parameters"];
        assert_eq!(params.get("$schema"), None);
        assert_eq!(params["properties"]["path"]["type"].as_str(), Some("STRING"));
    }

    #[test]
    fn test_map_messages_to_gemini_extracts_system_and_user_turns() {
        let messages = serde_json::json!([
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": "Hello!" }
        ]);
        let (system_instruction, contents) = map_messages_to_gemini(&messages, None, false, None).unwrap();
        assert!(system_instruction.is_some());
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[1].role, "model");
    }
}
