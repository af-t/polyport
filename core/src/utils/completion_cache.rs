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

/// Map an Anthropic stop reason to a Gemini finish reason for unified storage, mirroring TS
/// `mapAnthropicStopToGeminiFinish`. Returns `None` for `tool_use`/unknown so tool-call presence
/// drives the replay mapping instead.
#[must_use]
pub fn map_anthropic_stop_to_gemini_finish(reason: Option<&str>) -> Option<&'static str> {
    match reason {
        Some("max_tokens" | "model_context_window_exceeded") => Some("MAX_TOKENS"),
        Some("refusal") => Some("SAFETY"),
        Some("end_turn" | "stop_sequence" | "pause_turn") => Some("STOP"),
        _ => None,
    }
}

/// Build a completion entry from a collected Gemini turn, mirroring TS `populateCompletionCache`.
/// Returns `None` for empty turns (no text, reasoning, or tool calls), which TS does not store.
#[must_use]
pub fn gemini_completion_entry(
    collected: &crate::utils::sse::CollectedGemini,
    model: &str,
    created: u64,
) -> Option<CompletionEntry> {
    if collected.text.is_empty() && collected.reasoning.is_empty() && collected.function_calls.is_empty() {
        return None;
    }
    let token = crate::utils::history_cache::mint_reasoning_token(
        collected.thought_signature.as_deref().unwrap_or(""),
        &collected.reasoning,
    );
    Some(CompletionEntry {
        text: collected.text.clone(),
        reasoning: if collected.reasoning.is_empty() {
            None
        } else {
            Some(collected.reasoning.clone())
        },
        thought_signature: if token.is_empty() { None } else { Some(token) },
        finish_reason: collected.finish_reason.clone(),
        tool_calls: if collected.function_calls.is_empty() {
            None
        } else {
            Some(
                collected
                    .function_calls
                    .iter()
                    .map(|fc| CompletionToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                        name: fc.name.clone(),
                        arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                    })
                    .collect(),
            )
        },
        usage: CompletionUsage {
            input_tokens: u64::from(collected.prompt_tokens),
            output_tokens: u64::from(collected.output_tokens),
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
        },
        model: model.to_string(),
        created,
    })
}

/// Build a completion entry from a direct-Anthropic JSON response, mirroring TS `populateClaudeCache`.
/// Returns `None` for empty turns.
#[must_use]
pub fn claude_completion_entry(response: &Value, model: &str, created: u64) -> Option<CompletionEntry> {
    let blocks = response.get("content").and_then(Value::as_array);
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut signature = String::new();
    let mut tool_calls = Vec::new();
    if let Some(blocks) = blocks {
        for block in blocks {
            match block.get("type").and_then(Value::as_str).unwrap_or("") {
                "text" => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
                "thinking" => {
                    if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                        reasoning.push_str(t);
                    }
                    if let Some(s) = block.get("signature").and_then(Value::as_str) {
                        signature = s.to_string();
                    }
                }
                "tool_use" => {
                    tool_calls.push(CompletionToolCall {
                        id: block.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                        name: block.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                        arguments: block
                            .get("input")
                            .map(|v| serde_json::to_string(v).unwrap_or_default())
                            .unwrap_or_default(),
                    });
                }
                _ => {}
            }
        }
    }
    if text.is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
        return None;
    }
    let token = crate::utils::history_cache::mint_reasoning_token(&signature, &reasoning);
    let usage = response.get("usage");
    let num = |name: &str| usage.and_then(|u| u.get(name)).and_then(Value::as_u64).unwrap_or(0);
    Some(CompletionEntry {
        text,
        reasoning: if reasoning.is_empty() { None } else { Some(reasoning) },
        thought_signature: if token.is_empty() { None } else { Some(token) },
        finish_reason: map_anthropic_stop_to_gemini_finish(response.get("stop_reason").and_then(Value::as_str))
            .map(str::to_string),
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        usage: CompletionUsage {
            input_tokens: num("input_tokens"),
            output_tokens: num("output_tokens"),
            cache_read_tokens: num("cache_read_input_tokens"),
            cache_write_tokens: num("cache_creation_input_tokens"),
            reasoning_tokens: 0,
        },
        model: model.to_string(),
        created,
    })
}

/// Build a completion entry from a finished `OpenAI`-protocol Gemini stream state.
/// Returns `None` for unfinished or empty turns.
#[must_use]
pub fn gemini_stream_entry_from_openai(state: &crate::utils::sse::OpenAiStreamState) -> Option<CompletionEntry> {
    if !state.finished {
        return None;
    }
    gemini_completion_entry(
        &crate::utils::sse::CollectedGemini {
            text: state.full_text.clone(),
            reasoning: state.reasoning.clone(),
            function_calls: state.streamed_tools.clone(),
            thought_signature: if state.thought_signature.is_empty() {
                None
            } else {
                Some(state.thought_signature.clone())
            },
            finish_reason: state.finish_reason.clone(),
            prompt_tokens: state.prompt_tokens,
            output_tokens: state.output_tokens,
        },
        &state.model,
        state.created,
    )
}

/// Build a completion entry from a finished Anthropic-protocol Gemini stream state.
/// Returns `None` for unfinished or empty turns.
#[must_use]
pub fn gemini_stream_entry_from_anthropic(
    state: &crate::utils::sse::AnthropicStreamState,
    created: u64,
) -> Option<CompletionEntry> {
    if !state.finished {
        return None;
    }
    gemini_completion_entry(
        &crate::utils::sse::CollectedGemini {
            text: state.full_text.clone(),
            reasoning: state.reasoning.clone(),
            function_calls: state.streamed_tools.clone(),
            thought_signature: if state.thought_signature.is_empty() {
                None
            } else {
                Some(state.thought_signature.clone())
            },
            finish_reason: state.finish_reason.clone(),
            prompt_tokens: state.input_tokens,
            output_tokens: state.output_tokens,
        },
        &state.model,
        created,
    )
}

#[must_use]
pub fn codex_usage(usage: Option<&Value>) -> CompletionUsage {
    let usage = usage.unwrap_or(&Value::Null);
    let cached = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    CompletionUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_sub(cached),
        output_tokens: usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
        cache_read_tokens: cached,
        cache_write_tokens: 0,
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

/// Build a completion entry from an accumulated Codex turn, mirroring the TS Codex `cache.set`.
/// Returns `None` for unfinished or empty turns.
#[must_use]
pub fn codex_completion_entry(
    text: &str,
    acc: &crate::utils::sse::CodexAccumulator,
    model: &str,
    created: u64,
    finished: bool,
) -> Option<CompletionEntry> {
    if !finished || (text.is_empty() && acc.reasoning.is_empty() && acc.tools.is_empty()) {
        return None;
    }
    let token = if acc.encrypted.is_empty() {
        String::new()
    } else {
        crate::utils::history_cache::mint_reasoning_token(
            &serde_json::to_string(&acc.encrypted).unwrap_or_default(),
            &acc.reasoning,
        )
    };
    Some(CompletionEntry {
        text: text.to_string(),
        reasoning: if acc.reasoning.is_empty() {
            None
        } else {
            Some(acc.reasoning.clone())
        },
        thought_signature: if token.is_empty() { None } else { Some(token) },
        finish_reason: Some(if acc.tools.is_empty() {
            "stop".to_string()
        } else {
            "tool_calls".to_string()
        }),
        tool_calls: if acc.tools.is_empty() {
            None
        } else {
            Some(
                acc.tools
                    .iter()
                    .map(|t| CompletionToolCall {
                        id: t.id.clone(),
                        name: t.name.clone(),
                        arguments: t.args.clone(),
                    })
                    .collect(),
            )
        },
        usage: codex_usage(acc.usage.as_ref()),
        model: model.to_string(),
        created,
    })
}

/// Fetch a fresh full response for a request key.
#[must_use]
pub fn lookup_completion(key: &str) -> Option<CompletionEntry> {
    if key.is_empty() {
        return None;
    }
    COMPLETION_CACHE.lock().ok().and_then(|guard| guard.get_fresh(key))
}

fn usable_signature(sig: Option<&str>) -> Option<&str> {
    sig.filter(|s| !s.is_empty() && *s != super::history_cache::DUMMY_SIGNATURE)
}

/// Split text into 8-character replay words, mirroring TS `match(/.{1,8}/g)`.
#[must_use]
pub fn split_words(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(8).map(|c| c.iter().collect()).collect()
}

fn reasoning_details_block(format: &str, text: &str, signature: Option<&str>) -> Value {
    let mut entry = serde_json::json!({
        "type": "reasoning.text",
        "id": Value::Null,
        "text": text,
        "format": format,
        "index": 0
    });
    if let Some(sig) = usable_signature(signature) {
        entry["signature"] = Value::String(sig.to_string());
    }
    entry
}

/// `OpenAI` usage object mirroring TS `toOpenAIUsage`.
#[must_use]
pub fn openai_usage_value(usage: &CompletionUsage) -> Value {
    let prompt_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens;
    let mut prompt_details = serde_json::json!({
        "cached_tokens": usage.cache_read_tokens,
        "audio_tokens": 0
    });
    if usage.cache_write_tokens > 0 {
        prompt_details["cache_write_tokens"] = Value::from(usage.cache_write_tokens);
    }
    serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": prompt_tokens + usage.output_tokens,
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": {
            "reasoning_tokens": usage.reasoning_tokens,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    })
}

/// `Anthropic` usage object mirroring TS `toAnthropicUsage`.
#[must_use]
pub fn anthropic_usage_value(usage: &CompletionUsage) -> Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens
    })
}

/// `Anthropic` streaming delta usage mirroring TS `toAnthropicDeltaUsage`.
#[must_use]
pub fn anthropic_delta_usage_value(usage: &CompletionUsage) -> Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens
    })
}

/// Map a stored `Gemini` finish reason to `OpenAI`, mirroring TS `mapGeminiToOpenAIFinish`.
#[must_use]
pub fn map_gemini_to_openai_finish(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_calls";
    }
    match reason {
        Some("MAX_TOKENS") => "length",
        Some("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" | "LANGUAGE") => {
            "content_filter"
        }
        _ => "stop",
    }
}

/// Map a stored `Gemini` finish reason to `Anthropic`, mirroring TS `mapGeminiToAnthropicStop`.
#[must_use]
pub fn map_gemini_to_anthropic_stop(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_use";
    }
    match reason {
        Some("MAX_TOKENS") => "max_tokens",
        Some("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" | "LANGUAGE") => {
            "refusal"
        }
        _ => "end_turn",
    }
}

fn openai_chunk(id: &str, created: u64, model: &str, delta: &Value, finish_reason: &Value) -> String {
    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
    });
    format!("data: {chunk}\n\n")
}

/// SSE event stream replaying a cached entry as `OpenAI` chunks, mirroring TS completions HIT.
#[must_use]
pub fn replay_completions_stream(
    entry: &CompletionEntry,
    reasoning_format: &str,
    reasoning_excluded: bool,
    want_usage: bool,
) -> Vec<String> {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let mut events = Vec::new();
    let has_tools = entry.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
    if let Some(reasoning) = entry.reasoning.as_deref()
        && !reasoning.is_empty()
        && !reasoning_excluded
    {
        for word in split_words(reasoning) {
            let delta = serde_json::json!({
                "reasoning": word,
                "reasoning_content": word,
                "reasoning_details": [reasoning_details_block(reasoning_format, &word, None)]
            });
            events.push(openai_chunk(&id, entry.created, &entry.model, &delta, &Value::Null));
        }
    }
    if let Some(sig) = usable_signature(entry.thought_signature.as_deref())
        && !reasoning_excluded
    {
        let delta = serde_json::json!({
            "reasoning_details": [reasoning_details_block(reasoning_format, "", Some(sig))]
        });
        events.push(openai_chunk(&id, entry.created, &entry.model, &delta, &Value::Null));
    }
    if !entry.text.is_empty() {
        for word in split_words(&entry.text) {
            let delta = serde_json::json!({ "content": word });
            events.push(openai_chunk(&id, entry.created, &entry.model, &delta, &Value::Null));
        }
    }
    if let Some(tool_calls) = entry.tool_calls.as_deref() {
        for (index, tc) in tool_calls.iter().enumerate() {
            let delta = serde_json::json!({
                "tool_calls": [{
                    "index": index,
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": tc.arguments }
                }]
            });
            events.push(openai_chunk(&id, entry.created, &entry.model, &delta, &Value::Null));
        }
    }
    let finish = Value::String(map_gemini_to_openai_finish(entry.finish_reason.as_deref(), has_tools).to_string());
    let empty = serde_json::json!({});
    events.push(openai_chunk(&id, entry.created, &entry.model, &empty, &finish));
    if want_usage {
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": entry.created,
            "model": entry.model,
            "choices": [],
            "usage": openai_usage_value(&entry.usage)
        });
        events.push(format!("data: {chunk}\n\n"));
    }
    events.push("data: [DONE]\n\n".to_string());
    events
}

/// Non-streaming `OpenAI` completion replaying a cached entry, mirroring TS completions HIT.
#[must_use]
pub fn replay_completions_json(entry: &CompletionEntry, reasoning_format: &str, reasoning_excluded: bool) -> Value {
    let has_tools = entry.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": if entry.text.is_empty() { Value::Null } else { Value::String(entry.text.clone()) }
    });
    if let Some(reasoning) = entry.reasoning.as_deref()
        && !reasoning.is_empty()
        && !reasoning_excluded
    {
        message["reasoning"] = Value::String(reasoning.to_string());
        message["reasoning_content"] = Value::String(reasoning.to_string());
    }
    let sig = usable_signature(entry.thought_signature.as_deref());
    if (entry.reasoning.as_deref().is_some_and(|r| !r.is_empty()) || sig.is_some()) && !reasoning_excluded {
        message["reasoning_details"] = Value::Array(vec![reasoning_details_block(
            reasoning_format,
            entry.reasoning.as_deref().unwrap_or(""),
            sig,
        )]);
    }
    if let Some(tool_calls) = entry.tool_calls.as_deref() {
        message["tool_calls"] = Value::Array(
            tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": tc.arguments }
                    })
                })
                .collect(),
        );
    }
    serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": entry.created,
        "model": entry.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": map_gemini_to_openai_finish(entry.finish_reason.as_deref(), has_tools)
        }],
        "usage": openai_usage_value(&entry.usage)
    })
}

fn thinking_block_events(entry: &CompletionEntry, index: usize, reasoning: &str) -> Vec<String> {
    let mut events = vec![format!(
        "event: content_block_start\ndata: {}\n\n",
        serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "thinking", "thinking": "" }
        })
    )];
    for word in split_words(reasoning) {
        events.push(format!(
            "event: content_block_delta\ndata: {}\n\n",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "thinking_delta", "thinking": word }
            })
        ));
    }
    let sig = entry
        .thought_signature
        .clone()
        .unwrap_or_else(|| super::history_cache::DUMMY_SIGNATURE.to_string());
    events.push(format!(
        "event: content_block_delta\ndata: {}\n\n",
        serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "signature_delta", "signature": sig }
        })
    ));
    events.push(format!(
        "event: content_block_stop\ndata: {}\n\n",
        serde_json::json!({ "type": "content_block_stop", "index": index })
    ));
    events
}

fn text_block_events(text: &str, index: usize) -> Vec<String> {
    let mut events = vec![format!(
        "event: content_block_start\ndata: {}\n\n",
        serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "text", "text": "" }
        })
    )];
    for word in split_words(text) {
        events.push(format!(
            "event: content_block_delta\ndata: {}\n\n",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "text_delta", "text": word }
            })
        ));
    }
    events.push(format!(
        "event: content_block_stop\ndata: {}\n\n",
        serde_json::json!({ "type": "content_block_stop", "index": index })
    ));
    events
}

fn tool_block_events(tc: &CompletionToolCall, index: usize) -> Vec<String> {
    vec![
        format!(
            "event: content_block_start\ndata: {}\n\n",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "tool_use", "id": tc.id, "name": tc.name, "input": {} }
            })
        ),
        format!(
            "event: content_block_delta\ndata: {}\n\n",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "input_json_delta", "partial_json": tc.arguments }
            })
        ),
        format!(
            "event: content_block_stop\ndata: {}\n\n",
            serde_json::json!({ "type": "content_block_stop", "index": index })
        ),
    ]
}

/// SSE event stream replaying a cached entry as `Anthropic` events, mirroring TS messages HIT.
#[must_use]
pub fn replay_messages_stream(entry: &CompletionEntry, reasoning_excluded: bool) -> Vec<String> {
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4());
    let mut events = Vec::new();
    let has_tools = entry.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
    let mut start_usage = anthropic_usage_value(&entry.usage);
    start_usage["output_tokens"] = Value::from(0);
    events.push(format!(
        "event: message_start\ndata: {}\n\n",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": entry.model,
                "usage": start_usage
            }
        })
    ));
    let mut block_index = 0;
    if let Some(reasoning) = entry.reasoning.as_deref()
        && !reasoning.is_empty()
        && !reasoning_excluded
    {
        let thinking_index = block_index;
        block_index += 1;
        events.extend(thinking_block_events(entry, thinking_index, reasoning));
    }
    if !entry.text.is_empty() {
        let text_index = block_index;
        block_index += 1;
        events.extend(text_block_events(&entry.text, text_index));
    }
    if let Some(tool_calls) = entry.tool_calls.as_deref() {
        for tc in tool_calls {
            let tool_index = block_index;
            block_index += 1;
            events.extend(tool_block_events(tc, tool_index));
        }
    }
    events.push(format!(
        "event: message_delta\ndata: {}\n\n",
        serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": map_gemini_to_anthropic_stop(entry.finish_reason.as_deref(), has_tools),
                "stop_sequence": Value::Null
            },
            "usage": anthropic_delta_usage_value(&entry.usage)
        })
    ));
    events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());
    events
}

/// Non-streaming `Anthropic` message replaying a cached entry, mirroring TS messages HIT.
#[must_use]
pub fn replay_messages_json(entry: &CompletionEntry, reasoning_excluded: bool) -> Value {
    let has_tools = entry.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
    let mut blocks = Vec::new();
    if let Some(reasoning) = entry.reasoning.as_deref()
        && !reasoning.is_empty()
        && !reasoning_excluded
    {
        blocks.push(serde_json::json!({
            "type": "thinking",
            "thinking": reasoning,
            "signature": entry.thought_signature.clone()
                .unwrap_or_else(|| super::history_cache::DUMMY_SIGNATURE.to_string())
        }));
    }
    if !entry.text.is_empty() {
        blocks.push(serde_json::json!({ "type": "text", "text": entry.text }));
    }
    if let Some(tool_calls) = entry.tool_calls.as_deref() {
        for tc in tool_calls {
            let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
            blocks.push(serde_json::json!({ "type": "tool_use", "id": tc.id, "name": tc.name, "input": input }));
        }
    }
    serde_json::json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": entry.model,
        "stop_reason": map_gemini_to_anthropic_stop(entry.finish_reason.as_deref(), has_tools),
        "stop_sequence": Value::Null,
        "usage": anthropic_usage_value(&entry.usage)
    })
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
    fn test_finish_mappers_match_ts() {
        use super::{map_gemini_to_anthropic_stop, map_gemini_to_openai_finish};
        assert_eq!(map_gemini_to_openai_finish(Some("STOP"), false), "stop");
        assert_eq!(map_gemini_to_openai_finish(None, false), "stop");
        assert_eq!(map_gemini_to_openai_finish(Some("MAX_TOKENS"), false), "length");
        assert_eq!(map_gemini_to_openai_finish(Some("SAFETY"), false), "content_filter");
        assert_eq!(map_gemini_to_openai_finish(Some("RECITATION"), false), "content_filter");
        assert_eq!(map_gemini_to_openai_finish(Some("STOP"), true), "tool_calls");
        assert_eq!(map_gemini_to_openai_finish(Some("MAX_TOKENS"), true), "tool_calls");
        assert_eq!(map_gemini_to_anthropic_stop(Some("STOP"), false), "end_turn");
        assert_eq!(map_gemini_to_anthropic_stop(None, false), "end_turn");
        assert_eq!(map_gemini_to_anthropic_stop(Some("MAX_TOKENS"), false), "max_tokens");
        assert_eq!(map_gemini_to_anthropic_stop(Some("SAFETY"), false), "refusal");
        assert_eq!(map_gemini_to_anthropic_stop(Some("STOP"), true), "tool_use");
    }

    fn rich_entry() -> CompletionEntry {
        CompletionEntry {
            text: "hello world, this is a long reply".to_string(),
            reasoning: Some("thinking through the answer".to_string()),
            thought_signature: Some("sig_abc".to_string()),
            finish_reason: Some("STOP".to_string()),
            tool_calls: Some(vec![super::CompletionToolCall {
                id: "call_1".to_string(),
                name: "noop".to_string(),
                arguments: "{\"a\":1}".to_string(),
            }]),
            usage: CompletionUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: 2,
                cache_write_tokens: 0,
                reasoning_tokens: 4,
            },
            model: "m".to_string(),
            created: 1_700_000_001,
        }
    }

    fn parse_sse_data(event: &str) -> serde_json::Value {
        try_parse_sse_data(event).unwrap()
    }

    fn try_parse_sse_data(event: &str) -> Option<serde_json::Value> {
        let data_line = event.lines().find(|l| l.starts_with("data:"))?;
        serde_json::from_str(data_line.trim_start_matches("data:").trim()).ok()
    }

    #[test]
    fn test_replay_completions_stream_shape() {
        use super::replay_completions_stream;
        let events = replay_completions_stream(&rich_entry(), "google-gemini-v1", false, true);
        assert_eq!(events.last().unwrap(), "data: [DONE]\n\n");
        let usage = parse_sse_data(&events[events.len() - 2]);
        assert_eq!(usage.get("choices").unwrap().as_array().unwrap().len(), 0);
        assert_eq!(
            usage
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(Value::as_u64),
            Some(12)
        );
        assert_eq!(
            usage
                .get("usage")
                .and_then(|u| u.get("completion_tokens_details"))
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(Value::as_u64),
            Some(4)
        );
        let final_chunk = parse_sse_data(&events[events.len() - 3]);
        assert_eq!(
            final_chunk
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("finish_reason")),
            Some(&serde_json::Value::String("tool_calls".to_string()))
        );
        let first = parse_sse_data(&events[0]);
        let delta = first
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .unwrap();
        assert_eq!(delta.get("reasoning").and_then(Value::as_str), Some("thinking"));
        assert_eq!(
            delta
                .get("reasoning_details")
                .and_then(|d| d.get(0))
                .and_then(|d| d.get("format"))
                .and_then(Value::as_str),
            Some("google-gemini-v1")
        );
        let sig_chunk = events
            .iter()
            .map(|e| parse_sse_data(e))
            .find(|v| {
                v.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("reasoning_details"))
                    .is_some_and(|d| {
                        d.get(0).and_then(|b| b.get("signature")).and_then(Value::as_str) == Some("sig_abc")
                    })
            })
            .expect("signature chunk must carry the stored signature");
        let _ = sig_chunk;
        let content: String = events
            .iter()
            .filter_map(|e| try_parse_sse_data(e))
            .filter_map(|v| {
                v.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(content, "hello world, this is a long reply");
    }

    #[test]
    fn test_replay_completions_json_shape() {
        use super::replay_completions_json;
        let body = replay_completions_json(&rich_entry(), "google-gemini-v1", false);
        assert_eq!(body.get("object").and_then(Value::as_str), Some("chat.completion"));
        let choice = body.get("choices").and_then(|c| c.get(0)).unwrap();
        assert_eq!(choice.get("finish_reason").and_then(Value::as_str), Some("tool_calls"));
        let message = choice.get("message").unwrap();
        assert_eq!(
            message.get("reasoning").and_then(Value::as_str),
            Some("thinking through the answer")
        );
        assert_eq!(
            message
                .get("reasoning_details")
                .and_then(|d| d.get(0))
                .and_then(|d| d.get("signature"))
                .and_then(Value::as_str),
            Some("sig_abc")
        );
        let tool = message.get("tool_calls").and_then(|t| t.get(0)).unwrap();
        assert_eq!(tool.get("id").and_then(Value::as_str), Some("call_1"));
        assert_eq!(
            body.get("usage")
                .and_then(|u| u.get("total_tokens"))
                .and_then(Value::as_u64),
            Some(17)
        );
    }

    #[test]
    fn test_replay_messages_stream_shape() {
        use super::replay_messages_stream;
        let events = replay_messages_stream(&rich_entry(), false);
        assert!(events[0].starts_with("event: message_start\n"));
        let start = parse_sse_data(&events[0]);
        assert_eq!(
            start
                .get("message")
                .and_then(|m| m.get("usage"))
                .and_then(|u| u.get("output_tokens"))
                .and_then(Value::as_u64),
            Some(0)
        );
        assert!(events.iter().any(|e| e.contains("thinking_delta")));
        let sig_delta = events
            .iter()
            .map(|e| parse_sse_data(e))
            .find(|v| v.get("delta").and_then(|d| d.get("type")).and_then(Value::as_str) == Some("signature_delta"))
            .unwrap();
        assert_eq!(
            sig_delta
                .get("delta")
                .and_then(|d| d.get("signature"))
                .and_then(Value::as_str),
            Some("sig_abc")
        );
        assert!(events.iter().any(|e| e.contains("input_json_delta")));
        assert!(events.iter().any(|e| e.contains("call_1")));
        let delta = events
            .iter()
            .find(|e| e.starts_with("event: message_delta\n"))
            .map(|e| parse_sse_data(e))
            .unwrap();
        assert_eq!(
            delta
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(Value::as_str),
            Some("tool_use")
        );
        assert!(events.last().unwrap().starts_with("event: message_stop\n"));
    }

    #[test]
    fn test_replay_messages_json_shape() {
        use super::replay_messages_json;
        let body = replay_messages_json(&rich_entry(), false);
        assert_eq!(body.get("stop_reason").and_then(Value::as_str), Some("tool_use"));
        let blocks = body.get("content").and_then(Value::as_array).unwrap();
        assert_eq!(blocks[0].get("type").and_then(Value::as_str), Some("thinking"));
        assert_eq!(blocks[0].get("signature").and_then(Value::as_str), Some("sig_abc"));
        assert_eq!(blocks[1].get("type").and_then(Value::as_str), Some("text"));
        assert_eq!(blocks[2].get("type").and_then(Value::as_str), Some("tool_use"));
        assert_eq!(
            blocks[2].get("input").and_then(|i| i.get("a")).and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn test_replay_respects_reasoning_excluded() {
        use super::{replay_completions_json, replay_completions_stream, replay_messages_json, replay_messages_stream};
        assert!(
            replay_completions_stream(&rich_entry(), "google-gemini-v1", true, false)
                .iter()
                .all(|e| !e.contains("reasoning"))
        );
        let json = replay_completions_json(&rich_entry(), "google-gemini-v1", true);
        let message = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .unwrap();
        assert!(message.get("reasoning").is_none());
        assert!(message.get("reasoning_details").is_none());
        assert!(
            replay_messages_stream(&rich_entry(), true)
                .iter()
                .all(|e| !e.contains("thinking"))
        );
        let msg = replay_messages_json(&rich_entry(), true);
        assert!(msg.get("content").and_then(Value::as_array).is_some_and(|b| {
            b.iter()
                .all(|x| x.get("type").and_then(Value::as_str) != Some("thinking"))
        }));
    }

    #[test]
    fn test_anthropic_stop_mapper_matches_ts() {
        use super::map_anthropic_stop_to_gemini_finish;
        assert_eq!(
            map_anthropic_stop_to_gemini_finish(Some("max_tokens")),
            Some("MAX_TOKENS")
        );
        assert_eq!(
            map_anthropic_stop_to_gemini_finish(Some("model_context_window_exceeded")),
            Some("MAX_TOKENS")
        );
        assert_eq!(map_anthropic_stop_to_gemini_finish(Some("refusal")), Some("SAFETY"));
        assert_eq!(map_anthropic_stop_to_gemini_finish(Some("end_turn")), Some("STOP"));
        assert_eq!(map_anthropic_stop_to_gemini_finish(Some("stop_sequence")), Some("STOP"));
        assert_eq!(map_anthropic_stop_to_gemini_finish(Some("pause_turn")), Some("STOP"));
        assert_eq!(map_anthropic_stop_to_gemini_finish(Some("tool_use")), None);
        assert_eq!(map_anthropic_stop_to_gemini_finish(None), None);
    }

    #[test]
    fn test_gemini_entry_builder_gate_and_shape() {
        use super::gemini_completion_entry;
        use crate::utils::sse::{CollectedGemini, GeminiFunctionCall};
        let empty = CollectedGemini {
            text: String::new(),
            reasoning: String::new(),
            function_calls: Vec::new(),
            thought_signature: None,
            finish_reason: None,
            prompt_tokens: 0,
            output_tokens: 0,
        };
        assert!(gemini_completion_entry(&empty, "m", 99).is_none());
        let full = CollectedGemini {
            text: "hi".to_string(),
            reasoning: "why".to_string(),
            function_calls: vec![GeminiFunctionCall {
                name: "noop".to_string(),
                args: serde_json::json!({ "a": 1 }),
            }],
            thought_signature: Some("real_sig_pop".to_string()),
            finish_reason: Some("STOP".to_string()),
            prompt_tokens: 11,
            output_tokens: 7,
        };
        let entry = gemini_completion_entry(&full, "m", 99).unwrap();
        assert_eq!(entry.text, "hi");
        assert_eq!(entry.finish_reason.as_deref(), Some("STOP"));
        assert_eq!(entry.usage.input_tokens, 11);
        assert_eq!(entry.created, 99);
        let token = entry
            .thought_signature
            .expect("minted token must be stored, not the raw signature");
        assert!(token.starts_with("pxsig_"));
        assert!(!token.contains("real_sig_pop"));
        let tc = entry.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].name, "noop");
        assert_eq!(tc[0].arguments, "{\"a\":1}");
    }

    #[test]
    fn test_claude_entry_builder_parses_blocks() {
        use super::claude_completion_entry;
        let response = serde_json::json!({
            "content": [
                { "type": "thinking", "thinking": "plan", "signature": "claude_sig_1" },
                { "type": "text", "text": "done" },
                { "type": "tool_use", "id": "toolu_1", "name": "read", "input": { "p": "x" } }
            ],
            "stop_reason": "max_tokens",
            "usage": { "input_tokens": 20, "output_tokens": 9, "cache_read_input_tokens": 5 }
        });
        let entry = claude_completion_entry(&response, "c", 100).unwrap();
        assert_eq!(entry.text, "done");
        assert_eq!(entry.reasoning.as_deref(), Some("plan"));
        assert_eq!(entry.finish_reason.as_deref(), Some("MAX_TOKENS"));
        assert_eq!(entry.usage.input_tokens, 20);
        assert_eq!(entry.usage.cache_read_tokens, 5);
        let token = entry.thought_signature.unwrap();
        assert!(token.starts_with("pxsig_"));
        let tc = entry.tool_calls.unwrap();
        assert_eq!(tc[0].id, "toolu_1");
        assert_eq!(tc[0].arguments, "{\"p\":\"x\"}");

        let empty = serde_json::json!({ "content": [], "stop_reason": "end_turn" });
        assert!(claude_completion_entry(&empty, "c", 100).is_none());
    }

    #[test]
    fn test_stream_entry_gate_and_shape() {
        use super::{gemini_stream_entry_from_anthropic, gemini_stream_entry_from_openai};
        use crate::utils::sse::{AnthropicStreamState, OpenAiStreamState};
        let mut openai = OpenAiStreamState::new("chatcmpl-x", "m");
        openai.full_text = "streamed text".to_string();
        assert!(gemini_stream_entry_from_openai(&openai).is_none());
        openai.finished = true;
        openai.finish_reason = Some("MAX_TOKENS".to_string());
        openai.prompt_tokens = 6;
        openai.output_tokens = 2;
        let entry = gemini_stream_entry_from_openai(&openai).unwrap();
        assert_eq!(entry.text, "streamed text");
        assert_eq!(entry.finish_reason.as_deref(), Some("MAX_TOKENS"));
        assert_eq!(entry.usage.input_tokens, 6);
        assert_eq!(entry.model, "m");

        let mut anthropic = AnthropicStreamState::new("msg_x", "m");
        assert!(gemini_stream_entry_from_anthropic(&anthropic, 77).is_none());
        anthropic.finished = true;
        anthropic.full_text = "hi".to_string();
        let entry = gemini_stream_entry_from_anthropic(&anthropic, 77).unwrap();
        assert_eq!(entry.created, 77);
        assert_eq!(entry.text, "hi");
    }

    #[test]
    fn test_codex_entry_builder_gate_and_shape() {
        use super::codex_completion_entry;
        use crate::utils::sse::{CodexAccumulator, CodexToolAssembly};
        let empty = CodexAccumulator::default();
        assert!(codex_completion_entry("", &empty, "m", 5, true).is_none());
        assert!(codex_completion_entry("x", &empty, "m", 5, false).is_none());
        let acc = CodexAccumulator {
            tools: vec![CodexToolAssembly {
                id: "call_ab".to_string(),
                name: "run".to_string(),
                args: "{\"c\":\"ls\"}".to_string(),
            }],
            index_by_item: std::collections::HashMap::new(),
            reasoning: "plan".to_string(),
            encrypted: vec!["blob1".to_string(), "blob2".to_string()],
            usage: Some(serde_json::json!({
                "input_tokens": 30,
                "output_tokens": 12,
                "input_tokens_details": { "cached_tokens": 10 },
                "output_tokens_details": { "reasoning_tokens": 3 }
            })),
        };
        let entry = codex_completion_entry("out", &acc, "gpt", 55, true).unwrap();
        assert_eq!(entry.text, "out");
        assert_eq!(entry.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(entry.usage.input_tokens, 20);
        assert_eq!(entry.usage.cache_read_tokens, 10);
        assert_eq!(entry.usage.reasoning_tokens, 3);
        assert_eq!(entry.created, 55);
        let token = entry.thought_signature.expect("encrypted blobs must mint a token");
        assert!(token.starts_with("pxsig_"));
        let tc = entry.tool_calls.unwrap();
        assert_eq!(tc[0].id, "call_ab");
        assert_eq!(tc[0].arguments, "{\"c\":\"ls\"}");
    }

    #[test]
    fn test_codex_collected_populates_and_replays_tool_turn() {
        use super::{
            codex_completion_entry, lookup_completion, replay_completions_json, replay_messages_json,
            write_completion_cache,
        };
        use crate::utils::sse::collect_codex_response;
        let frames = [
            serde_json::json!({ "type": "response.output_text.delta", "delta": "I will check." }),
            serde_json::json!({
                "type": "response.output_item.done",
                "item": { "type": "function_call", "id": "f1", "call_id": "call_1", "name": "get_weather", "arguments": "{\"loc\":\"Tokyo\"}" }
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": { "usage": { "input_tokens": 50, "output_tokens": 20 } }
            }),
        ];
        let collected = collect_codex_response(&frames);
        let entry = codex_completion_entry(&collected.text, &collected.acc, "gpt-5.6-luna", 0, collected.finished)
            .expect("turn must build completion entry");
        let key = format!("codex-buffered-test-{}", std::process::id());
        write_completion_cache(&key, entry);

        let cached = lookup_completion(&key).expect("cached entry must exist");
        assert_eq!(cached.text, "I will check.");
        assert_eq!(cached.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(cached.tool_calls.as_ref().map(Vec::len), Some(1));

        let openai_json = replay_completions_json(&cached, "openai-codex-v1", false);
        assert_eq!(openai_json["choices"][0]["finish_reason"], "tool_calls");
        let tc = &openai_json["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], "{\"loc\":\"Tokyo\"}");

        let anthropic_json = replay_messages_json(&cached, false);
        assert_eq!(anthropic_json["stop_reason"], "tool_use");
        let tool_block = anthropic_json["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("anthropic replay must have tool_use block");
        assert_eq!(tool_block["id"], "call_1");
        assert_eq!(tool_block["name"], "get_weather");
        assert_eq!(tool_block["input"]["loc"], "Tokyo");
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
