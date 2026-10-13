use serde_json::Value;
use sha2::{Digest, Sha256};

const TOKEN_PREFIX: &str = "pxsig_";

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantAnchor {
    pub text: Option<String>,
    pub token: Option<String>,
}

impl AssistantAnchor {
    fn empty() -> Self {
        Self {
            text: None,
            token: None,
        }
    }

    fn complete(&self) -> bool {
        self.text.is_some() && self.token.is_some()
    }
}

#[must_use]
pub fn normalize_reply_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hex_of(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

fn tail_key(text: Option<&str>) -> Option<String> {
    let normalized = normalize_reply_text(text?);
    if normalized.is_empty() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let digest = hasher.finalize();
    Some(format!("tail:{}", hex_of(&digest[..20])))
}

#[must_use]
pub fn lookup_keys(header: Option<&str>, token: Option<&str>, tail_text: Option<&str>) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(value) = header.filter(|s| !s.is_empty()) {
        keys.push(format!("hdr:{value}"));
    }
    if let Some(value) = token.filter(|s| !s.is_empty()) {
        keys.push(format!("tok:{value}"));
    }
    if let Some(tail) = tail_key(tail_text) {
        keys.push(tail);
    }
    keys
}

#[must_use]
pub fn record_keys(header: Option<&str>, minted_token: Option<&str>, reply_text: Option<&str>) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(value) = header.filter(|s| !s.is_empty()) {
        keys.push(format!("hdr:{value}"));
    }
    if let Some(value) = minted_token.filter(|s| !s.is_empty()) {
        keys.push(format!("tok:{value}"));
    }
    if let Some(tail) = tail_key(reply_text) {
        keys.push(tail);
    }
    keys
}

fn text_of_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text")?.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn token_from_reasoning(msg: &Value) -> Option<String> {
    if let Some(details) = msg.get("reasoning_details") {
        let items: Vec<&Value> = match details {
            Value::Array(list) => list.iter().collect(),
            single => vec![single],
        };
        for detail in items {
            if let Some(sig) = detail.get("signature").and_then(Value::as_str)
                && sig.starts_with(TOKEN_PREFIX)
            {
                return Some(sig.to_owned());
            }
        }
    }
    if let Some(Value::Array(parts)) = msg.get("content") {
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("thinking")
                && let Some(sig) = part.get("signature").and_then(Value::as_str)
                && sig.starts_with(TOKEN_PREFIX)
            {
                return Some(sig.to_owned());
            }
        }
    }
    None
}

#[must_use]
pub fn last_assistant_from_messages(messages: &Value) -> AssistantAnchor {
    let mut anchor = AssistantAnchor::empty();
    let Value::Array(items) = messages else {
        return anchor;
    };
    for msg in items.iter().rev() {
        if anchor.complete() {
            break;
        }
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if anchor.token.is_none() {
            anchor.token = token_from_reasoning(msg);
        }
        if anchor.text.is_none() {
            let text = text_of_content(msg.get("content"));
            if !text.is_empty() {
                anchor.text = Some(text);
            }
        }
    }
    anchor
}

#[must_use]
pub fn last_assistant_from_gemini(contents: &Value) -> AssistantAnchor {
    let mut anchor = AssistantAnchor::empty();
    let Value::Array(items) = contents else {
        return anchor;
    };
    for entry in items.iter().rev() {
        if anchor.complete() {
            break;
        }
        if entry.get("role").and_then(Value::as_str) != Some("model") {
            continue;
        }
        let Some(Value::Array(parts)) = entry.get("parts") else {
            continue;
        };
        if anchor.token.is_none() {
            anchor.token = parts.iter().find_map(|part| {
                part.get("thoughtSignature")
                    .and_then(Value::as_str)
                    .filter(|sig| sig.starts_with(TOKEN_PREFIX))
                    .map(str::to_owned)
            });
        }
        if anchor.text.is_none() {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text")?.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if !text.is_empty() {
                anchor.text = Some(text);
            }
        }
    }
    anchor
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn whitespace_collapses_and_trims() {
        assert_eq!(normalize_reply_text("  a\n\n  b  "), "a b");
    }

    #[test]
    fn lookup_keys_ordered_hdr_tok_tail() {
        let keys = lookup_keys(Some("h1"), Some("pxsig_x"), Some("hello"));
        assert_eq!(keys[0], "hdr:h1");
        assert_eq!(keys[1], "tok:pxsig_x");
        assert_eq!(keys[2], record_keys(None, None, Some("hello"))[0]);
    }

    #[test]
    fn record_tail_equals_next_turn_lookup_tail() {
        let rec = record_keys(None, None, Some("Done.  "));
        let look = lookup_keys(None, None, Some("Done."));
        assert_eq!(rec, look);
    }

    #[test]
    fn empty_inputs_produce_no_keys() {
        assert_eq!(lookup_keys(None, None, None), Vec::<String>::new());
        assert_eq!(record_keys(None, None, Some("   ")), Vec::<String>::new());
    }

    #[test]
    fn tail_anchor_is_case_sensitive() {
        let lower = record_keys(None, None, Some("done."));
        let upper = record_keys(None, None, Some("Done."));
        assert_eq!(lower.len(), 1);
        assert_ne!(lower, upper);
    }

    #[test]
    fn messages_last_assistant_text_and_token() {
        let msgs = json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "answer",
              "reasoning_details": [{ "signature": "pxsig_abc" }] },
            { "role": "user", "content": "more" }
        ]);
        assert_eq!(
            last_assistant_from_messages(&msgs),
            AssistantAnchor {
                text: Some("answer".to_owned()),
                token: Some("pxsig_abc".to_owned())
            }
        );
    }

    #[test]
    fn messages_anthropic_thinking_block_signature() {
        let msgs = json!([
            { "role": "assistant", "content": [
                { "type": "thinking", "signature": "pxsig_anthropic_sig" },
                { "type": "text", "text": "hello" }
            ] }
        ]);
        assert_eq!(
            last_assistant_from_messages(&msgs),
            AssistantAnchor {
                text: Some("hello".to_owned()),
                token: Some("pxsig_anthropic_sig".to_owned())
            }
        );
    }

    #[test]
    fn messages_multiple_text_parts_skip_invalid() {
        let msgs = json!([
            { "role": "assistant", "content": [
                { "type": "text", "text": "part1" },
                { "type": "image", "source": "..." },
                null,
                { "type": "text", "text": "part2" }
            ] }
        ]);
        assert_eq!(
            last_assistant_from_messages(&msgs),
            AssistantAnchor {
                text: Some("part1 part2".to_owned()),
                token: None
            }
        );
    }

    #[test]
    fn messages_trailing_tool_call_falls_back() {
        let msgs = json!([
            { "role": "user", "content": "q" },
            { "role": "assistant", "content": "answer",
              "reasoning_details": [{ "signature": "pxsig_abc" }] },
            { "role": "tool", "content": "result" },
            { "role": "assistant", "content": "",
              "tool_calls": [{ "id": "t1", "function": { "name": "x", "arguments": "{}" } }] },
            { "role": "user", "content": "next" }
        ]);
        assert_eq!(
            last_assistant_from_messages(&msgs),
            AssistantAnchor {
                text: Some("answer".to_owned()),
                token: Some("pxsig_abc".to_owned())
            }
        );
    }

    #[test]
    fn messages_token_and_text_from_different_turns() {
        let msgs = json!([
            { "role": "assistant", "content": "older text" },
            { "role": "assistant", "content": "",
              "reasoning_details": [{ "signature": "pxsig_new" }] }
        ]);
        assert_eq!(
            last_assistant_from_messages(&msgs),
            AssistantAnchor {
                text: Some("older text".to_owned()),
                token: Some("pxsig_new".to_owned())
            }
        );
    }

    #[test]
    fn messages_invalid_inputs_give_empty_anchor() {
        let empty = AssistantAnchor {
            text: None,
            token: None,
        };
        assert_eq!(last_assistant_from_messages(&json!(null)), empty);
        assert_eq!(last_assistant_from_messages(&json!([])), empty);
        assert_eq!(
            last_assistant_from_messages(&json!([{ "role": "user", "content": "hi" }])),
            empty
        );
        assert_eq!(
            last_assistant_from_messages(&json!([{ "role": "assistant", "content": "" }])),
            empty
        );
        assert_eq!(
            last_assistant_from_messages(&json!([{ "role": "assistant", "content": [] }])),
            empty
        );
        assert_eq!(
            last_assistant_from_messages(&json!([{ "role": "assistant", "content": null }])),
            empty
        );
    }

    #[test]
    fn gemini_last_model_text_and_token() {
        let contents = json!([
            { "role": "user", "parts": [{ "text": "hi" }] },
            { "role": "model", "parts": [{ "text": "ans" }, { "thoughtSignature": "pxsig_z" }] },
            { "role": "user", "parts": [{ "text": "more" }] }
        ]);
        assert_eq!(
            last_assistant_from_gemini(&contents),
            AssistantAnchor {
                text: Some("ans".to_owned()),
                token: Some("pxsig_z".to_owned())
            }
        );
    }

    #[test]
    fn gemini_multiple_parts_skip_non_text() {
        let contents = json!([
            { "role": "model", "parts": [
                { "text": "part1" },
                { "functionCall": { "name": "fn" } },
                { "text": "part2", "thoughtSignature": "pxsig_gemini_sig" }
            ] }
        ]);
        assert_eq!(
            last_assistant_from_gemini(&contents),
            AssistantAnchor {
                text: Some("part1 part2".to_owned()),
                token: Some("pxsig_gemini_sig".to_owned())
            }
        );
    }

    #[test]
    fn gemini_trailing_tool_call_falls_back() {
        let contents = json!([
            { "role": "model", "parts": [{ "text": "ans" }, { "thoughtSignature": "pxsig_z" }] },
            { "role": "user", "parts": [{ "functionResponse": { "name": "fn" } }] },
            { "role": "model", "parts": [{ "functionCall": { "name": "fn2" } }] }
        ]);
        assert_eq!(
            last_assistant_from_gemini(&contents),
            AssistantAnchor {
                text: Some("ans".to_owned()),
                token: Some("pxsig_z".to_owned())
            }
        );
    }

    #[test]
    fn gemini_invalid_inputs_give_empty_anchor() {
        let empty = AssistantAnchor {
            text: None,
            token: None,
        };
        assert_eq!(last_assistant_from_gemini(&json!(null)), empty);
        assert_eq!(last_assistant_from_gemini(&json!([])), empty);
        assert_eq!(
            last_assistant_from_gemini(&json!([{ "role": "user", "parts": [{ "text": "hi" }] }])),
            empty
        );
        assert_eq!(
            last_assistant_from_gemini(&json!([{ "role": "model", "parts": [] }])),
            empty
        );
        assert_eq!(
            last_assistant_from_gemini(&json!([{ "role": "model", "parts": null }])),
            empty
        );
    }
}
