use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};
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
        self.insert_at(key, value, now_secs());
    }

    fn insert_at(&mut self, key: String, value: T, created: u64) {
        if !self.map.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.map.insert(key, (value, created));
        while self.map.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn get_fresh(&self, key: &str) -> Option<T> {
        let (value, created) = self.map.get(key)?;
        if !is_fresh(*created) {
            return None;
        }
        Some(value.clone())
    }

    fn iter(&self) -> impl Iterator<Item = (&String, &(T, u64))> {
        self.map.iter()
    }
}

const MAX_CACHE_SIZE: usize = 1000;
const CACHE_TTL_SECS: u64 = 24 * 60 * 60;
const SAVE_INTERVAL_SECS: u64 = 30;

static HISTORY_CACHE: LazyLock<Mutex<BoundedMap<HistoryEntry>>> = LazyLock::new(|| {
    let mut map = BoundedMap::new(MAX_CACHE_SIZE);
    for (key, entry, created) in load_history_file(&cache_dir().join("history-cache.json")) {
        map.insert_at(key, entry, created);
    }
    Mutex::new(map)
});
static SIGNATURE_CACHE: LazyLock<Mutex<BoundedMap<SignatureEntry>>> = LazyLock::new(|| {
    let mut map = BoundedMap::new(MAX_CACHE_SIZE);
    for (key, entry, created) in load_signature_file(&cache_dir().join("signature-cache.json")) {
        map.insert_at(key, entry, created);
    }
    Mutex::new(map)
});
static LAST_SAVE: LazyLock<Mutex<Option<std::time::Instant>>> = LazyLock::new(|| Mutex::new(None));

#[must_use]
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir).join("polyport");
    }
    #[cfg(test)]
    {
        return std::env::temp_dir().join("polyport-test-cache");
    }
    #[allow(unreachable_code)]
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache")
        .join("polyport")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryFileEntry {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reasoning: String,
    #[serde(default, skip_serializing_if = "String::is_empty", rename = "thoughtSignature")]
    thought_signature: String,
    #[serde(default)]
    created: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignatureFileEntry {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reasoning: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    signature: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    provider: String,
    #[serde(default)]
    created: u64,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn is_fresh(created: u64) -> bool {
    created == 0 || now_secs().saturating_sub(created) <= CACHE_TTL_SECS
}

fn load_history_file(path: &Path) -> Vec<(String, HistoryEntry, u64)> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(raw) = serde_json::from_str::<HashMap<String, HistoryFileEntry>>(&content) else {
        return Vec::new();
    };
    raw.into_iter()
        .filter(|(_, e)| is_usable_signature(&e.thought_signature) && is_fresh(e.created))
        .map(|(k, e)| {
            let created = e.created;
            (
                k,
                HistoryEntry {
                    thought_signature: e.thought_signature,
                    reasoning: e.reasoning,
                },
                created,
            )
        })
        .collect()
}

fn load_signature_file(path: &Path) -> Vec<(String, SignatureEntry, u64)> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(raw) = serde_json::from_str::<HashMap<String, SignatureFileEntry>>(&content) else {
        return Vec::new();
    };
    raw.into_iter()
        .filter(|(_, e)| !e.signature.is_empty() && is_fresh(e.created))
        .map(|(k, e)| {
            let created = e.created;
            (
                k,
                SignatureEntry {
                    signature: e.signature,
                    reasoning: e.reasoning,
                },
                created,
            )
        })
        .collect()
}

pub fn save_caches_to(dir: &Path) {
    let histories: Vec<(String, HistoryEntry, u64)> = HISTORY_CACHE
        .lock()
        .map(|guard| guard.iter().map(|(k, (v, c))| (k.clone(), v.clone(), *c)).collect())
        .unwrap_or_default();
    let signatures: Vec<(String, SignatureEntry, u64)> = SIGNATURE_CACHE
        .lock()
        .map(|guard| guard.iter().map(|(k, (v, c))| (k.clone(), v.clone(), *c)).collect())
        .unwrap_or_default();
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let history_map: HashMap<String, HistoryFileEntry> = histories
        .into_iter()
        .map(|(k, v, created)| {
            (
                k,
                HistoryFileEntry {
                    reasoning: v.reasoning,
                    thought_signature: v.thought_signature,
                    created,
                },
            )
        })
        .collect();
    if let Ok(json) = serde_json::to_string(&history_map) {
        let _ = std::fs::write(dir.join("history-cache.json"), json);
    }
    let signature_map: HashMap<String, SignatureFileEntry> = signatures
        .into_iter()
        .map(|(k, v, created)| {
            (
                k,
                SignatureFileEntry {
                    reasoning: v.reasoning,
                    signature: v.signature,
                    provider: "gemini".to_string(),
                    created,
                },
            )
        })
        .collect();
    if let Ok(json) = serde_json::to_string(&signature_map) {
        let _ = std::fs::write(dir.join("signature-cache.json"), json);
    }
}

fn maybe_persist() {
    let due = LAST_SAVE
        .lock()
        .is_ok_and(|guard| guard.is_none_or(|t| t.elapsed().as_secs() >= SAVE_INTERVAL_SECS));
    if !due {
        return;
    }
    if let Ok(mut guard) = LAST_SAVE.lock() {
        *guard = Some(std::time::Instant::now());
    }
    save_caches_to(&cache_dir());
}

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
    maybe_persist();
}

#[must_use]
pub fn lookup_tool_signature(tool_id: &str) -> Option<HistoryEntry> {
    if tool_id.is_empty() {
        return None;
    }
    HISTORY_CACHE
        .lock()
        .ok()
        .and_then(|guard| guard.get_fresh(tool_id))
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
    maybe_persist();
    token
}

#[must_use]
pub fn resolve_reasoning_token(token: &str) -> Option<ResolvedSignature> {
    if !is_usable_signature(token) || !token.starts_with("pxsig_") {
        return None;
    }
    SIGNATURE_CACHE.lock().ok().and_then(|guard| {
        guard.get_fresh(token).and_then(|entry| {
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
    maybe_persist();
}

#[must_use]
pub fn lookup_history(key: &str) -> Option<HistoryEntry> {
    if key.is_empty() {
        return None;
    }
    HISTORY_CACHE
        .lock()
        .ok()
        .and_then(|guard| guard.get_fresh(key))
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
    fn test_save_and_load_round_trip() {
        record_tool_signature("test-disk-id-1", "pxsig_disk_abc", "disk reasoning");
        let token = mint_reasoning_token("real_sig_disk", "disk thought");
        let dir = std::env::temp_dir().join("polyport-test-disk-1");
        save_caches_to(&dir);
        let histories = load_history_file(&dir.join("history-cache.json"));
        let found = histories.iter().find(|(k, _, _)| k == "test-disk-id-1").unwrap();
        assert_eq!(found.1.thought_signature, "pxsig_disk_abc");
        assert_eq!(found.1.reasoning, "disk reasoning");
        let sigs = load_signature_file(&dir.join("signature-cache.json"));
        let resolved = sigs.iter().find(|(k, _, _)| k == &token).unwrap();
        assert_eq!(resolved.1.signature, "real_sig_disk");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_skips_expired_and_malformed() {
        let dir = std::env::temp_dir().join("polyport-test-disk-2");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("history-cache.json"),
            serde_json::json!({
                "fresh": { "thoughtSignature": "sig_fresh", "reasoning": "r", "created": 9_999_999_999u64 },
                "stale": { "thoughtSignature": "sig_stale", "reasoning": "r", "created": 1u64 }
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("signature-cache.json"), "not-json{{{").unwrap();
        let histories = load_history_file(&dir.join("history-cache.json"));
        assert!(histories.iter().any(|(k, _, _)| k == "fresh"));
        assert!(!histories.iter().any(|(k, _, _)| k == "stale"));
        assert_eq!(load_signature_file(&dir.join("signature-cache.json")).len(), 0);
        assert_eq!(load_history_file(&dir.join("does-not-exist.json")).len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cache_dir_defaults_under_polyport() {
        let dir = cache_dir();
        let leaf = dir.file_name().and_then(std::ffi::OsStr::to_str);
        if std::env::var_os("XDG_CACHE_HOME").is_none() {
            assert_eq!(leaf, Some("polyport-test-cache"));
        } else {
            assert_eq!(leaf, Some("polyport"));
        }
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
