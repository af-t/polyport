use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    pub thought_signature: String,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct HistoryContext {
    pub model: String,
    pub system: Value,
    pub tools: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureEntry {
    pub signature: String,
    pub reasoning: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSignature {
    pub signature: String,
    pub reasoning: String,
}

struct BoundedMap<T> {
    map: HashMap<String, (T, u64)>,
    order: VecDeque<String>,
    capacity: usize,
}

impl<T: Clone> BoundedMap<T> {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn insert(&mut self, key: String, value: T) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if !self.map.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.map.insert(key, (value, now));
        while self.map.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn get_fresh(&self, key: &str, ttl_secs: u64) -> Option<T> {
        let (value, created) = self.map.get(key)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if now.saturating_sub(*created) > ttl_secs {
            return None;
        }
        Some(value.clone())
    }
}

const MAX_CACHE_SIZE: usize = 1000;
const CACHE_TTL_SECS: u64 = 24 * 60 * 60;

static HISTORY_CACHE: LazyLock<Mutex<BoundedMap<HistoryEntry>>> =
    LazyLock::new(|| Mutex::new(BoundedMap::new(MAX_CACHE_SIZE)));
static SIGNATURE_CACHE: LazyLock<Mutex<BoundedMap<SignatureEntry>>> =
    LazyLock::new(|| Mutex::new(BoundedMap::new(MAX_CACHE_SIZE)));

pub const DUMMY_SIGNATURE: &str = "dummy_sig_from_proxy";
pub const FALLBACK_SIGNATURE: &str = "context_engineering_is_the_way_to_go";

fn is_usable_signature(sig: &str) -> bool {
    !sig.is_empty() && sig != DUMMY_SIGNATURE
}

pub fn record_tool_signature(tool_id: &str, signature: &str, reasoning: &str) {
    if tool_id.is_empty() || !is_usable_signature(signature) {
        return;
    }
    if let Ok(mut guard) = HISTORY_CACHE.lock() {
        guard.insert(
            tool_id.to_string(),
            HistoryEntry {
                thought_signature: signature.to_string(),
                reasoning: reasoning.to_string(),
            },
        );
    }
}

#[must_use]
pub fn lookup_tool_signature(tool_id: &str) -> Option<HistoryEntry> {
    if tool_id.is_empty() {
        return None;
    }
    HISTORY_CACHE
        .lock()
        .ok()
        .and_then(|guard| guard.get_fresh(tool_id, CACHE_TTL_SECS))
        .filter(|entry| is_usable_signature(&entry.thought_signature))
}

#[must_use]
pub fn mint_reasoning_token(signature: &str, reasoning: &str) -> String {
    if !is_usable_signature(signature) {
        return String::new();
    }
    let digest = Sha256::digest(format!("gemini\n{signature}").as_bytes());
    let token = format!("pxsig_{:.40}", sha256_hex_bytes(&digest));
    if let Ok(mut guard) = SIGNATURE_CACHE.lock() {
        guard.insert(
            token.clone(),
            SignatureEntry {
                signature: signature.to_string(),
                reasoning: reasoning.to_string(),
            },
        );
    }
    token
}

#[must_use]
pub fn resolve_reasoning_token(token: &str) -> Option<ResolvedSignature> {
    if !is_usable_signature(token) || !token.starts_with("pxsig_") {
        return None;
    }
    SIGNATURE_CACHE.lock().ok().and_then(|guard| {
        guard.get_fresh(token, CACHE_TTL_SECS).and_then(|entry| {
            if entry.signature.is_empty() {
                return None;
            }
            Some(ResolvedSignature {
                signature: entry.signature,
                reasoning: entry.reasoning,
            })
        })
    })
}

pub fn record_history(key: &str, signature: &str, reasoning: &str) {
    if key.is_empty() || !is_usable_signature(signature) {
        return;
    }
    if let Ok(mut guard) = HISTORY_CACHE.lock() {
        guard.insert(
            key.to_string(),
            HistoryEntry {
                thought_signature: signature.to_string(),
                reasoning: reasoning.to_string(),
            },
        );
    }
}

#[must_use]
pub fn lookup_history(key: &str) -> Option<HistoryEntry> {
    if key.is_empty() {
        return None;
    }
    HISTORY_CACHE
        .lock()
        .ok()
        .and_then(|guard| guard.get_fresh(key, CACHE_TTL_SECS))
        .filter(|entry| is_usable_signature(&entry.thought_signature))
}

fn deterministic_stringify(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.to_string(),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(deterministic_stringify).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", Value::String((*k).clone()), deterministic_stringify(&map[*k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn normalize_message_for_history_key(m: &Value) -> Value {
    let Some(obj) = m.as_object() else {
        return m.clone();
    };
    let mut normalized = serde_json::Map::new();
    if let Some(role) = obj.get("role") {
        normalized.insert("role".to_string(), role.clone());
    }
    for key in ["name", "tool_call_id"] {
        if let Some(v) = obj.get(key) {
            normalized.insert(key.to_string(), v.clone());
        }
    }
    if let Some(content) = obj.get("content") {
        if content.is_string() {
            normalized.insert("content".to_string(), content.clone());
        } else if let Some(blocks) = content.as_array() {
            let norm_blocks: Vec<Value> = blocks
                .iter()
                .map(|block| {
                    let Some(b) = block.as_object() else {
                        return block.clone();
                    };
                    let block_type = b.get("type").and_then(Value::as_str).unwrap_or("");
                    match block_type {
                        "text" => serde_json::json!({ "type": "text", "text": b.get("text") }),
                        "tool_use" => serde_json::json!({
                            "type": "tool_use",
                            "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input")
                        }),
                        "tool_result" => serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": b.get("tool_use_id"),
                            "content": b.get("content")
                        }),
                        _ => {
                            let mut copy = b.clone();
                            copy.remove("cache_control");
                            copy.remove("cacheControl");
                            Value::Object(copy)
                        }
                    }
                })
                .collect();
            normalized.insert("content".to_string(), Value::Array(norm_blocks));
        }
    }
    if let Some(tool_calls) = obj.get("tool_calls").and_then(Value::as_array) {
        let norm_calls: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                let Some(t) = tc.as_object() else {
                    return tc.clone();
                };
                let mut norm = serde_json::Map::new();
                for key in ["id", "type"] {
                    if let Some(v) = t.get(key) {
                        norm.insert(key.to_string(), v.clone());
                    }
                }
                if let Some(f) = t.get("function").and_then(Value::as_object) {
                    let mut norm_fn = serde_json::Map::new();
                    for key in ["name", "arguments"] {
                        if let Some(v) = f.get(key) {
                            norm_fn.insert(key.to_string(), v.clone());
                        }
                    }
                    norm.insert("function".to_string(), Value::Object(norm_fn));
                }
                Value::Object(norm)
            })
            .collect();
        normalized.insert("tool_calls".to_string(), Value::Array(norm_calls));
    }
    Value::Object(normalized)
}

fn sha256_hex_bytes(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn sha256_hex(input: &str) -> String {
    sha256_hex_bytes(&Sha256::digest(input.as_bytes()))
}

#[must_use]
pub fn history_key_chain(model: &str, messages: &Value, system: &Value, tools: &Value) -> Vec<String> {
    let seed = serde_json::json!({ "model": model, "system": system, "tools": tools });
    let mut hash = sha256_hex(&deterministic_stringify(&seed));
    let mut chain = vec![hash.clone()];
    if let Some(arr) = messages.as_array() {
        for m in arr {
            let normalized = normalize_message_for_history_key(m);
            hash = sha256_hex(&format!("{hash} {}", deterministic_stringify(&normalized)));
            chain.push(hash.clone());
        }
    }
    chain
}

#[must_use]
pub fn history_key(model: &str, messages: &Value, system: &Value, tools: &Value) -> String {
    history_key_chain(model, messages, system, tools)
        .last()
        .cloned()
        .unwrap_or_default()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_lookup_round_trip() {
        record_tool_signature("test-cache-id-1", "pxsig_live_abc", "some reasoning");
        let entry = lookup_tool_signature("test-cache-id-1").unwrap();
        assert_eq!(entry.thought_signature, "pxsig_live_abc");
        assert_eq!(entry.reasoning, "some reasoning");
    }

    #[test]
    fn test_dummy_and_empty_signatures_are_dropped() {
        record_tool_signature("test-cache-id-2", DUMMY_SIGNATURE, "");
        record_tool_signature("test-cache-id-3", "", "");
        assert_eq!(lookup_tool_signature("test-cache-id-2"), None);
        assert_eq!(lookup_tool_signature("test-cache-id-3"), None);
        assert_eq!(lookup_tool_signature("test-cache-id-unknown"), None);
    }

    #[test]
    fn test_mint_and_resolve_round_trip() {
        let token = mint_reasoning_token("real_sig_xyz", "thought text");
        assert!(token.starts_with("pxsig_"));
        let resolved = resolve_reasoning_token(&token).unwrap();
        assert_eq!(resolved.signature, "real_sig_xyz");
        assert_eq!(resolved.reasoning, "thought text");
    }

    #[test]
    fn test_mint_rejects_empty_and_dummy() {
        assert_eq!(mint_reasoning_token("", "reasoning"), "");
        assert_eq!(mint_reasoning_token(DUMMY_SIGNATURE, "reasoning"), "");
        assert_eq!(resolve_reasoning_token(""), None);
        assert_eq!(resolve_reasoning_token(DUMMY_SIGNATURE), None);
        assert_eq!(resolve_reasoning_token("pxsig_unknown"), None);
    }

    #[test]
    fn test_history_key_round_trip() {
        let messages = serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "hello" }
        ]);
        let key = history_key("model-x", &messages, &serde_json::Value::Null, &serde_json::Value::Null);
        record_history(&key, "pxsig_hist", "prior thought");
        let entry = lookup_history(&key).unwrap();
        assert_eq!(entry.thought_signature, "pxsig_hist");

        let chain = history_key_chain("model-x", &messages, &serde_json::Value::Null, &serde_json::Value::Null);
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[2], key);
    }
}
