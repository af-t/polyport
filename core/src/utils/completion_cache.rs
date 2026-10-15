use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::history_cache::{cache_dir, sha256_hex};

pub const COMPLETION_CACHE_TTL_SECS: u64 = 24 * 60 * 60;
const COMPLETION_CACHE_CAP: usize = 1000;
const SAVE_INTERVAL_SECS: u64 = 30;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionUsage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: u64,
    #[serde(rename = "outputTokens")]
    pub output_tokens: u64,
    #[serde(rename = "cacheReadTokens", default)]
    pub cache_read_tokens: u64,
    #[serde(rename = "cacheWriteTokens", default)]
    pub cache_write_tokens: u64,
    #[serde(rename = "reasoningTokens", default)]
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionEntry {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(rename = "finishReason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(rename = "toolCalls", skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<CompletionToolCall>>,
    pub usage: CompletionUsage,
    pub model: String,
    pub created: u64,
}

struct CompletionMap {
    order: VecDeque<String>,
    entries: HashMap<String, (CompletionEntry, u64)>,
}

impl CompletionMap {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn insert(&mut self, key: String, entry: CompletionEntry, created_ms: u64) {
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, (entry, created_ms));
        while self.order.len() > COMPLETION_CACHE_CAP {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn get_fresh(&self, key: &str) -> Option<CompletionEntry> {
        let (entry, created_ms) = self.entries.get(key)?;
        if now_ms().saturating_sub(*created_ms) >= COMPLETION_CACHE_TTL_SECS * 1000 {
            return None;
        }
        Some(entry.clone())
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        u64::try_from(d.as_millis().min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
    })
}

fn created_to_ms(created: u64) -> u64 {
    if created < 1_000_000_000_000 {
        created.saturating_mul(1000)
    } else {
        created
    }
}

fn completion_file(dir: &Path) -> PathBuf {
    dir.join("completions-cache.json")
}

fn load_completion_file(path: &Path) -> Vec<(String, CompletionEntry, u64)> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    if text.trim().is_empty() {
        return Vec::new();
    }
    let parsed: serde_json::Map<String, Value> = serde_json::from_str(&text).unwrap_or_default();
    let now = now_ms();
    let mut out = Vec::new();
    for (key, val) in parsed {
        let Ok(entry) = serde_json::from_value::<CompletionEntry>(val) else {
            continue;
        };
        let created_ms = created_to_ms(entry.created);
        if now.saturating_sub(created_ms) < COMPLETION_CACHE_TTL_SECS * 1000 {
            out.push((key, entry, created_ms));
        }
    }
    out
}

static COMPLETION_CACHE: LazyLock<Mutex<CompletionMap>> = LazyLock::new(|| {
    let mut map = CompletionMap::new();
    for (key, entry, created_ms) in load_completion_file(&completion_file(&cache_dir())) {
        map.insert(key, entry, created_ms);
    }
    Mutex::new(map)
});

static LAST_SAVE: LazyLock<Mutex<Option<Instant>>> = LazyLock::new(|| Mutex::new(None));

fn save_completions_to(dir: &Path) {
    let snapshot: Vec<(String, CompletionEntry)> = COMPLETION_CACHE
        .lock()
        .map(|guard| guard.entries.iter().map(|(k, (e, _))| (k.clone(), e.clone())).collect())
        .unwrap_or_default();
    let mut data = serde_json::Map::with_capacity(snapshot.len());
    for (key, entry) in snapshot {
        data.insert(key, serde_json::to_value(&entry).unwrap_or(Value::Null));
    }
    let path = completion_file(dir);
    if let Some(parent) = path.parent()
        && std::fs::create_dir_all(parent).is_ok()
    {
        let tmp = path.with_extension("json.tmp");
        if serde_json::to_string_pretty(&Value::Object(data))
            .map_err(|_| ())
            .and_then(|text| std::fs::write(&tmp, text).map_err(|_| ()))
            .is_ok()
        {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

fn maybe_persist() {
    let due = LAST_SAVE
        .lock()
        .is_ok_and(|guard| guard.is_none_or(|t| t.elapsed().as_secs() >= SAVE_INTERVAL_SECS));
    if !due {
        return;
    }
    save_completions_to(&cache_dir());
    if let Ok(mut guard) = LAST_SAVE.lock() {
        *guard = Some(Instant::now());
    }
}

/// Store a full upstream response under its request key, stamping seconds when absent.
pub fn write_completion_cache(key: &str, mut entry: CompletionEntry) {
    if key.is_empty() {
        return;
    }
    if entry.created == 0 {
        entry.created = now_secs();
    }
    let created_ms = created_to_ms(entry.created);
    if let Ok(mut guard) = COMPLETION_CACHE.lock() {
        guard.insert(key.to_string(), entry, created_ms);
    }
    maybe_persist();
}

/// Fetch a fresh full response for a request key.
#[must_use]
pub fn lookup_completion(key: &str) -> Option<CompletionEntry> {
    if key.is_empty() {
        return None;
    }
    COMPLETION_CACHE.lock().ok().and_then(|guard| guard.get_fresh(key))
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

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn fold_chain(seed: &Value, items: &[Value]) -> Vec<String> {
    let mut head = sha256_hex(&deterministic_stringify(seed));
    let mut chain = vec![head.clone()];
    for item in items {
        head = sha256_hex(&format!("{head} {}", deterministic_stringify(item)));
        chain.push(head.clone());
    }
    chain
}

/// Full-payload cache key mirroring TS `getCacheKeyChain`.
#[must_use]
pub fn completion_key_chain(body: &Value, route_fingerprint: Option<&str>) -> Vec<String> {
    let obj = body.as_object();
    let get = |name: &str| obj.and_then(|o| o.get(name));
    let mut constant = serde_json::Map::new();
    constant.insert(
        "model".to_string(),
        Value::String(get("model").and_then(Value::as_str).unwrap_or("").to_string()),
    );
    let system = get("system")
        .filter(|v| !v.is_null())
        .cloned()
        .unwrap_or(Value::String(String::new()));
    constant.insert("system".to_string(), system);
    let temperature = get("temperature")
        .filter(|v| !v.is_null())
        .cloned()
        .unwrap_or(Value::from(0.7));
    constant.insert("temperature".to_string(), temperature);
    let max_tokens = ["max_tokens", "max_output_tokens", "max_completion_tokens"]
        .iter()
        .filter_map(|name| get(name))
        .find(|v| is_truthy(v))
        .cloned();
    if let Some(v) = max_tokens {
        constant.insert("max_tokens".to_string(), v);
    }
    for name in [
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
        "stop",
        "tools",
        "tool_choice",
        "thinking",
        "seed",
        "reasoning",
        "reasoning_effort",
        "output_config",
        "thinking_budget",
    ] {
        if let Some(v) = get(name) {
            constant.insert(name.to_string(), v.clone());
        }
    }
    if let Some(fp) = route_fingerprint {
        constant.insert("routeFingerprint".to_string(), Value::String(fp.to_string()));
    }
    let messages: Vec<Value> = get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
    fold_chain(&Value::Object(constant), &messages)
}

/// Full-payload cache key mirroring TS `getCacheKey` (last chain link).
#[must_use]
pub fn completion_key(body: &Value, route_fingerprint: Option<&str>) -> String {
    let chain = completion_key_chain(body, route_fingerprint);
    chain.last().cloned().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        COMPLETION_CACHE_TTL_SECS, CompletionEntry, CompletionUsage, completion_key, completion_key_chain,
        load_completion_file, lookup_completion, write_completion_cache,
    };

    fn sample_entry(created: u64) -> CompletionEntry {
        CompletionEntry {
            text: "hello".to_string(),
            reasoning: Some("because".to_string()),
            thought_signature: Some("sig123".to_string()),
            finish_reason: Some("STOP".to_string()),
            tool_calls: None,
            usage: CompletionUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: 0,
            },
            model: "m".to_string(),
            created,
        }
    }

    fn body(json: serde_json::Value) -> serde_json::Value {
        json
    }

    #[test]
    fn test_key_matches_ts_basic_completions() {
        let key = completion_key(
            &body(serde_json::json!({
                "model": "gemini-3.8-flash-tiered",
                "messages": [{ "role": "user", "content": "hi" }]
            })),
            None,
        );
        assert_eq!(key, "e5d970bbdb94e5b70456b4bbc1b9a3fb8195d41b0bc665fc85dd09442f345e49");
    }

    #[test]
    fn test_key_matches_ts_with_tools_temp() {
        let key = completion_key(
            &body(serde_json::json!({
                "model": "gemini-3.8-flash-tiered",
                "temperature": 0.2,
                "messages": [{ "role": "user", "content": "hi" }],
                "tools": [{ "type": "function", "function": { "name": "noop", "parameters": { "type": "object" } } }]
            })),
            None,
        );
        assert_eq!(key, "1af4a7ad2720d331b94c66eb69fe2de83ddadf4538c51c0d9ab2274fa2643ad4");
    }

    #[test]
    fn test_key_matches_ts_with_fingerprint() {
        let key = completion_key(
            &body(serde_json::json!({ "model": "x", "messages": [{ "role": "user", "content": "hi" }] })),
            Some("openai|gemini:gemini-3.8-flash"),
        );
        assert_eq!(key, "e7669bdb688f48ceeda7e36787270f231b4cb6380aa3866eacb5d54f5ab242f6");
    }

    #[test]
    fn test_key_matches_ts_effort() {
        let key = completion_key(
            &body(serde_json::json!({
                "model": "m",
                "reasoning_effort": "high",
                "messages": [{ "role": "system", "content": "s" }, { "role": "user", "content": "q" }]
            })),
            None,
        );
        assert_eq!(key, "7b7a219bc916b03d64fe922d73cae043307af34269d28390cc51ab6f50797d07");
    }

    #[test]
    fn test_key_matches_ts_messages_shape() {
        let key = completion_key(
            &body(serde_json::json!({
                "model": "claude-sonnet-5-native",
                "system": "sys",
                "max_tokens": 100,
                "thinking": { "type": "enabled" },
                "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hey" }] }]
            })),
            None,
        );
        assert_eq!(key, "5f7a8b74f9f23fb4c148662d3ace4095718675beb6e391906633bd42dc52a360");
    }

    #[test]
    fn test_chain_length_matches_message_count_plus_one() {
        let chain = completion_key_chain(
            &body(serde_json::json!({
                "model": "m",
                "messages": [{ "role": "user", "content": "a" }, { "role": "user", "content": "b" }]
            })),
            None,
        );
        assert_eq!(chain.len(), 3);
        assert_eq!(
            chain.last().unwrap(),
            &completion_key(
                &body(serde_json::json!({
                    "model": "m",
                    "messages": [{ "role": "user", "content": "a" }, { "role": "user", "content": "b" }]
                })),
                None,
            )
        );
    }

    #[test]
    fn test_completion_file_round_trip_and_filters() {
        let dir = std::env::temp_dir().join(format!("polyport-compl-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let stale = now_secs.saturating_sub(COMPLETION_CACHE_TTL_SECS + 60);
        let mut data = serde_json::Map::new();
        data.insert(
            "fresh".to_string(),
            serde_json::to_value(sample_entry(now_secs)).unwrap(),
        );
        data.insert(
            "fresh-ms".to_string(),
            serde_json::to_value(CompletionEntry {
                created: now_secs * 1000,
                ..sample_entry(now_secs)
            })
            .unwrap(),
        );
        data.insert("stale".to_string(), serde_json::to_value(sample_entry(stale)).unwrap());
        data.insert(
            "old-shape".to_string(),
            serde_json::json!({
                "text": "x", "model": "m", "created": now_secs,
                "usage": { "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4 }
            }),
        );
        data.insert("garbage".to_string(), serde_json::json!({ "nope": true }));
        std::fs::write(
            dir.join("completions-cache.json"),
            serde_json::to_string(&data).unwrap(),
        )
        .unwrap();

        let loaded = load_completion_file(&dir.join("completions-cache.json"));
        let keys: Vec<&str> = loaded.iter().map(|(k, _, _)| k.as_str()).collect();
        assert!(keys.contains(&"fresh"));
        assert!(keys.contains(&"fresh-ms"));
        assert!(!keys.contains(&"stale"));
        assert!(!keys.contains(&"old-shape"));
        assert!(!keys.contains(&"garbage"));
        let fresh = loaded.iter().find(|(k, _, _)| k == "fresh").unwrap();
        assert_eq!(fresh.1.usage.input_tokens, 10);
        assert_eq!(fresh.1.thought_signature.as_deref(), Some("sig123"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_completion_disk_shape_uses_ts_field_names() {
        let value = serde_json::to_value(sample_entry(1_700_000_000)).unwrap();
        assert_eq!(value.get("thoughtSignature").and_then(|v| v.as_str()), Some("sig123"));
        assert_eq!(value.get("finishReason").and_then(|v| v.as_str()), Some("STOP"));
        assert_eq!(
            value
                .get("usage")
                .and_then(|u| u.get("inputTokens"))
                .and_then(Value::as_u64),
            Some(10)
        );
        assert!(value.get("thought_signature").is_none());
    }

    #[test]
    fn test_write_and_lookup_completion_round_trip() {
        let key = format!("test-compl-{}", std::process::id());
        write_completion_cache(&key, sample_entry(0));
        let found = lookup_completion(&key).expect("written entry must be readable");
        assert_eq!(found.text, "hello");
        assert!(found.created > 0);
        assert!(lookup_completion("").is_none());
    }

    #[test]
    fn test_zero_temperature_and_max_tokens_fall_through_like_ts() {
        let with_zero = completion_key(
            &body(serde_json::json!({ "model": "m", "temperature": 0, "max_tokens": 0, "messages": [] })),
            None,
        );
        let without = completion_key(
            &body(serde_json::json!({ "model": "m", "temperature": 0, "messages": [] })),
            None,
        );
        assert_eq!(with_zero, without);
    }
}
