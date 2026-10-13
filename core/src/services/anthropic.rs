use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::config::get_port;
use crate::services::account_pool::{AccountPool, AttemptError};
use crate::services::credentials::{
    ClaudeAccount, ClaudeOauth, derive_claude_account_id, get_claude_accounts, upsert_claude_account,
};
use crate::types::{ModelClass, Provider};
use crate::utils::anthropic_session::{get_deterministic_session_id, get_device_id};
use crate::utils::rate_limit::{
    NonAccountError, RateLimitError, UpstreamError, is_rate_limit_status, parse_cooldown_ms,
};

pub const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const CLAUDE_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
pub const CLAUDE_CODE_BILLING_MARKER: &str = "x-anthropic-billing-header:";
pub const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages?beta=true";

pub const CLAUDE_CODE_BETAS: &[&str] = &[
    "claude-code-20250219",
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "thinking-token-count-2026-05-13",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "mid-conversation-system-2026-04-07",
    "advisor-tool-2026-03-01",
    "effort-2025-11-24",
    "extended-cache-ttl-2025-04-11",
];

#[must_use]
pub fn get_claude_code_version() -> String {
    std::env::var("POLYPORT_CLAUDE_CLI_VERSION").unwrap_or_else(|_| "2.1.181".to_string())
}

#[must_use]
pub fn get_claude_code_user_agent() -> String {
    format!("claude-cli/{} (external, cli)", get_claude_code_version())
}

#[must_use]
pub fn map_anthropic_model(requested_model: &str) -> String {
    let lower = requested_model.to_lowercase();
    if lower.contains("fable") {
        return "claude-fable-5".to_string();
    }
    if let Some(stripped) = lower.strip_suffix("-native") {
        return stripped.to_string();
    }
    if lower.contains("opus-5") || lower.contains("opus5") {
        return "claude-opus-5".to_string();
    }
    if lower.contains("opus-4-8") || lower.contains("opus-4.8") {
        return "claude-opus-4-8".to_string();
    }
    if lower.contains("opus-4-7") || lower.contains("opus-4.7") {
        return "claude-opus-4-7".to_string();
    }
    if lower.contains("opus-4-6") || lower.contains("opus-4.6") {
        return "claude-opus-4-6".to_string();
    }
    if lower.contains("opus") {
        return "claude-opus-5".to_string();
    }
    if lower.contains("haiku") {
        return "claude-haiku-4-5-20251001".to_string();
    }
    if lower.contains("sonnet-5") || lower.contains("sonnet5") {
        return "claude-sonnet-5".to_string();
    }
    "claude-sonnet-4-6".to_string()
}

#[must_use]
pub fn normalize_anthropic_effort(effort: Option<&str>) -> Option<String> {
    let raw = effort?.trim().to_lowercase();
    if raw == "minimal" {
        return Some("low".to_string());
    }
    if matches!(raw.as_str(), "low" | "medium" | "high" | "xhigh" | "max") {
        Some(raw)
    } else {
        None
    }
}

pub fn apply_thinking_constraints(body: &mut Value) {
    let is_enabled = body
        .get("thinking")
        .is_some_and(|t| t.get("type").and_then(Value::as_str) != Some("disabled"));
    if is_enabled {
        let Some(obj) = body.as_object_mut() else {
            return;
        };
        obj.insert("temperature".to_string(), Value::from(1));
        obj.remove("top_p");
        obj.remove("top_k");
    }
}

fn extract_system_texts(system: &Value) -> Vec<String> {
    match system {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|b| {
                if let Value::String(s) = b {
                    Some(s.clone())
                } else if b.get("type").and_then(Value::as_str) == Some("text") {
                    b.get("text").and_then(Value::as_str).map(ToString::to_string)
                } else {
                    None
                }
            })
            .collect(),
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("text") {
                map.get("text")
                    .and_then(Value::as_str)
                    .map(|s| vec![s.to_string()])
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

#[must_use]
pub fn with_claude_code_identity(system: Option<&Value>) -> Vec<Value> {
    if let Some(sys) = system {
        let texts = extract_system_texts(sys);
        if texts
            .iter()
            .any(|t| t.trim() == CLAUDE_CODE_IDENTITY || t.starts_with(CLAUDE_CODE_BILLING_MARKER))
        {
            return match sys {
                Value::Array(arr) => arr.clone(),
                Value::String(s) => vec![serde_json::json!({ "type": "text", "text": s })],
                other => vec![other.clone()],
            };
        }
    }

    let mut blocks = vec![serde_json::json!({ "type": "text", "text": CLAUDE_CODE_IDENTITY })];
    if let Some(sys) = system {
        match sys {
            Value::String(s) => {
                if !s.trim().is_empty() {
                    blocks.push(serde_json::json!({ "type": "text", "text": s }));
                }
            }
            Value::Array(arr) => blocks.extend(arr.iter().cloned()),
            other => blocks.push(other.clone()),
        }
    }
    blocks
}

#[must_use]
pub fn merge_anthropic_betas(client_beta: Option<&str>) -> String {
    let mut betas: Vec<String> = CLAUDE_CODE_BETAS.iter().map(|s| (*s).to_string()).collect();
    if let Some(cb) = client_beta {
        for part in cb.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() && !betas.iter().any(|b| b == trimmed) {
                betas.push(trimmed.to_string());
            }
        }
    }
    betas.join(",")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTtl {
    FiveMinutes,
    OneHour,
    Off,
}

#[must_use]
pub fn get_cache_ttl() -> CacheTtl {
    let val = std::env::var("POLYPORT_CACHE_TTL")
        .unwrap_or_else(|_| "5m".to_string())
        .to_lowercase();
    match val.as_str() {
        "off" | "none" | "false" | "0" => CacheTtl::Off,
        "1h" | "60m" => CacheTtl::OneHour,
        _ => CacheTtl::FiveMinutes,
    }
}

fn block_has_cache_control(b: &Value) -> bool {
    b.is_object() && b.get("cache_control").is_some()
}

fn body_has_any_cache_control(body: &Value) -> bool {
    let sys_has = body
        .get("system")
        .and_then(Value::as_array)
        .is_some_and(|sys| sys.iter().any(block_has_cache_control));
    if sys_has {
        return true;
    }
    let tools_has = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| t.iter().any(block_has_cache_control));
    if tools_has {
        return true;
    }
    if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
        for m in msgs {
            let parts_has = m
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|p| p.iter().any(block_has_cache_control));
            if parts_has {
                return true;
            }
        }
    }
    false
}

fn mark_last_block(arr: &mut [Value], budget: &mut usize, marker: &Value) {
    if *budget == 0 || arr.is_empty() {
        return;
    }
    let Some(last) = arr.last_mut() else {
        return;
    };
    let Some(obj) = last.as_object_mut() else {
        return;
    };
    obj.insert("cache_control".to_string(), marker.clone());
    *budget -= 1;
}

pub fn inject_claude_code_cache_control(body: &mut Value, ttl: CacheTtl) {
    if ttl == CacheTtl::Off || !body.is_object() || body_has_any_cache_control(body) {
        return;
    }

    let marker = if ttl == CacheTtl::OneHour {
        serde_json::json!({ "type": "ephemeral", "ttl": "1h" })
    } else {
        serde_json::json!({ "type": "ephemeral" })
    };

    let mut budget = 4usize;
    if let Some(sys) = body.get_mut("system").and_then(Value::as_array_mut) {
        mark_last_block(sys, &mut budget, &marker);
    } else if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
        mark_last_block(tools, &mut budget, &marker);
    }

    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        let mut user_indices = Vec::new();
        for (i, m) in msgs.iter().enumerate().rev() {
            if m.get("role").and_then(Value::as_str) == Some("user") {
                user_indices.push(i);
                if user_indices.len() == 2 {
                    break;
                }
            }
        }

        for idx in user_indices {
            if budget == 0 {
                break;
            }
            let msg = &mut msgs[idx];
            if let Some(content_str) = msg.get("content").and_then(Value::as_str) {
                let text_obj = serde_json::json!({ "type": "text", "text": content_str, "cache_control": marker });
                msg["content"] = serde_json::json!([text_obj]);
                budget -= 1;
            } else if let Some(parts) = msg.get_mut("content").and_then(Value::as_array_mut) {
                mark_last_block(parts, &mut budget, &marker);
            }
        }
    }
}

#[must_use]
pub fn build_claude_code_headers(
    token: &str,
    anthropic_version: Option<&str>,
    client_beta: Option<&str>,
    client_user_agent: Option<&str>,
) -> HeaderMap {
    let mut map = HeaderMap::new();
    let ua = match client_user_agent {
        Some(s) if s.starts_with("claude-cli/") => s.to_string(),
        _ => get_claude_code_user_agent(),
    };

    let insert = |h: &mut HeaderMap, k: &'static str, v: &str| {
        if let Ok(val) = HeaderValue::from_str(v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };

    insert(&mut map, "content-type", "application/json");
    insert(&mut map, "accept", "application/json");
    insert(&mut map, "anthropic-version", anthropic_version.unwrap_or("2023-06-01"));
    insert(&mut map, "anthropic-beta", &merge_anthropic_betas(client_beta));
    insert(&mut map, "anthropic-dangerous-direct-browser-access", "true");
    insert(&mut map, "x-app", "cli");
    insert(&mut map, "user-agent", &ua);
    insert(&mut map, "authorization", &format!("Bearer {token}"));

    insert(&mut map, "x-stainless-lang", "js");
    insert(&mut map, "x-stainless-package-version", "0.94.0");
    insert(
        &mut map,
        "x-stainless-os",
        if cfg!(target_os = "macos") {
            "MacOS"
        } else if cfg!(target_os = "windows") {
            "Windows"
        } else {
            "Linux"
        },
    );
    insert(
        &mut map,
        "x-stainless-arch",
        if cfg!(target_arch = "x86_64") {
            "x64"
        } else if cfg!(target_arch = "aarch64") {
            "arm64"
        } else {
            "Other"
        },
    );
    insert(&mut map, "x-stainless-runtime", "node");
    insert(&mut map, "x-stainless-runtime-version", "v24.3.0");
    insert(&mut map, "x-stainless-retry-count", "0");
    insert(&mut map, "x-stainless-timeout", "600");

    map
}

pub fn stamp_claude_metadata(body: &mut Value, session_id: &str) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let metadata = obj.entry("metadata").or_insert_with(|| serde_json::json!({}));
    if metadata.get("user_id").is_none() {
        let user_id = serde_json::json!({
            "device_id": get_device_id(),
            "account_uuid": "",
            "session_id": session_id,
        });
        metadata["user_id"] = Value::String(user_id.to_string());
    }
}

static PENDING_VERIFIERS: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

fn base64url_encode(data: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() * 4).div_ceil(3));
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i];
        let b1 = if i + 1 < data.len() { data[i + 1] } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] } else { 0 };

        out.push(CHARSET[(b0 >> 2) as usize] as char);
        out.push(CHARSET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if i + 1 < data.len() {
            out.push(CHARSET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        }
        if i + 2 < data.len() {
            out.push(CHARSET[(b2 & 0x3f) as usize] as char);
        }
        i += 3;
    }
    out
}

fn generate_pkce() -> (String, String) {
    let u1 = uuid::Uuid::new_v4();
    let u2 = uuid::Uuid::new_v4();
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(u1.as_bytes());
    bytes[16..].copy_from_slice(u2.as_bytes());

    let verifier = base64url_encode(&bytes);
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = base64url_encode(&hasher.finalize());
    (verifier, challenge)
}

fn url_encode(input: &str) -> String {
    use std::fmt::Write;
    let mut encoded = String::new();
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| (d.as_secs() * 1000 + u64::from(d.subsec_millis())).cast_signed())
}

#[must_use]
pub fn get_claude_login_url(port: u16) -> String {
    let (verifier, challenge) = generate_pkce();
    if let Ok(mut lock) = PENDING_VERIFIERS.lock() {
        if lock.len() >= 100 {
            lock.pop_front();
        }
        lock.push_back(verifier.clone());
    }

    let redirect_uri = url_encode(&format!("http://localhost:{port}/callback"));
    let scope = url_encode(
        "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload",
    );

    format!(
        "https://claude.com/cai/oauth/authorize?code=true&client_id={CLAUDE_CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}&scope={scope}&code_challenge={challenge}&code_challenge_method=S256&state={verifier}"
    )
}

#[must_use]
pub fn get_stored_code_verifier(verifier: Option<&str>) -> Option<String> {
    let mut lock = PENDING_VERIFIERS.lock().ok()?;
    if let Some(v) = verifier {
        if let Some(pos) = lock.iter().position(|x| x == v) {
            return lock.remove(pos);
        }
        return None;
    }
    lock.pop_back()
}

pub fn set_stored_code_verifier(verifier: Option<&str>) {
    if let Ok(mut lock) = PENDING_VERIFIERS.lock() {
        if let Some(v) = verifier {
            lock.push_back(v.to_string());
        } else {
            lock.clear();
        }
    }
}

/// # Errors
/// Fails if saving to credential storage fails.
pub fn save_claude_token_data(
    account_id: &str,
    access_token: &str,
    refresh_token: &str,
    expires_at: i64,
) -> anyhow::Result<()> {
    upsert_claude_account(&ClaudeAccount {
        id: account_id.to_string(),
        claude_ai_oauth: ClaudeOauth {
            access_token: access_token.to_string(),
            refresh_token: Some(refresh_token.to_string()),
            expires_at: Some(expires_at),
        },
    })
}

#[derive(Debug, Deserialize)]
struct ClaudeTokenExchangeResponse {
    access_token: String,
    refresh_token: String,
    expires_in: Option<u64>,
    account: Option<ClaudeAccountInfo>,
}

#[derive(Debug, Deserialize)]
struct ClaudeAccountInfo {
    email_address: Option<String>,
    uuid: Option<String>,
}

/// # Errors
/// Fails if verifier is missing, exchange returns non-200, or payload is malformed.
pub async fn exchange_claude_code(
    code: &str,
    state: Option<&str>,
    client: Option<&reqwest::Client>,
) -> anyhow::Result<()> {
    let verifier = get_stored_code_verifier(state)
        .ok_or_else(|| anyhow::anyhow!("Session expired or code verifier not found. Please try logging in again."))?;

    let default_client = reqwest::Client::new();
    let http = client.unwrap_or(&default_client);
    let port = get_port();

    let payload = serde_json::json!({
        "grant_type": "authorization_code",
        "client_id": CLAUDE_CLIENT_ID,
        "redirect_uri": format!("http://localhost:{port}/callback"),
        "code": code,
        "code_verifier": verifier,
        "state": verifier,
    });

    let res = http
        .post(CLAUDE_TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !res.status().is_success() {
        let err_text = res.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Anthropic OAuth exchange failed: {err_text}"));
    }

    let data: ClaudeTokenExchangeResponse = res.json().await?;
    let expires_in_ms = data.expires_in.unwrap_or(3600).cast_signed() * 1000;
    let expires_at = now_epoch_ms() + expires_in_ms;

    let email = data.account.as_ref().and_then(|a| a.email_address.as_deref());
    let uuid = data.account.as_ref().and_then(|a| a.uuid.as_deref());
    let account_id = derive_claude_account_id(email, uuid)?;

    save_claude_token_data(&account_id, &data.access_token, &data.refresh_token, expires_at)
}

#[derive(Debug, Deserialize)]
struct ClaudeTokenRefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

async fn try_refresh_claude_token(http: &reqwest::Client, acc_id: &str, refresh_token: &str) -> Option<String> {
    let payload = serde_json::json!({
        "grant_type": "refresh_token",
        "client_id": CLAUDE_CLIENT_ID,
        "refresh_token": refresh_token,
    });
    let res = http
        .post(CLAUDE_TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        return None;
    }
    let data: ClaudeTokenRefreshResponse = res.json().await.ok()?;
    let new_refresh = data.refresh_token.as_deref().unwrap_or(refresh_token);
    let expires_in_ms = data.expires_in.unwrap_or(3600).cast_signed() * 1000;
    let new_expires_at = now_epoch_ms() + expires_in_ms;
    let _ = save_claude_token_data(acc_id, &data.access_token, new_refresh, new_expires_at);
    Some(data.access_token)
}

/// Retrieves an existing Claude token, refreshing it if within 15 minutes of expiration.
///
/// # Errors
/// Returns an error if refreshing fails and no existing token can be used.
pub async fn get_claude_token(
    account: Option<&ClaudeAccount>,
    client: Option<&reqwest::Client>,
) -> anyhow::Result<Option<String>> {
    let default_accounts = get_claude_accounts();
    let acc = account.or_else(|| default_accounts.first());

    let Some(acc) = acc else {
        return Ok(None);
    };

    let oauth = &acc.claude_ai_oauth;
    if oauth.access_token.is_empty() {
        return Ok(None);
    }

    let now = now_epoch_ms();
    let expires_at = oauth.expires_at.unwrap_or(0);
    let is_expiring = expires_at > 0 && (expires_at - now) < 15 * 60 * 1000;

    let refresh_token = match (is_expiring, &oauth.refresh_token) {
        (true, Some(rt)) => Some(rt.as_str()),
        _ => None,
    };

    if let Some(rt) = refresh_token {
        let default_client = reqwest::Client::new();
        let http = client.unwrap_or(&default_client);
        if let Some(new_token) = try_refresh_claude_token(http, &acc.id, rt).await {
            return Ok(Some(new_token));
        }
    }

    Ok(Some(oauth.access_token.clone()))
}

#[derive(Debug, Clone)]
pub struct CallClaudeParams {
    pub body: Value,
    pub model_class: ModelClass,
    pub preferred_account_id: Option<String>,
    pub client_headers: Option<HeaderMap>,
}

#[derive(Debug)]
pub struct ClaudeStreamResult {
    pub response: reqwest::Response,
    pub account_id: String,
}

async fn resolve_claude_candidates(
    pool: &Arc<RwLock<AccountPool>>,
    accounts: &[ClaudeAccount],
    preferred: Option<&str>,
    class: ModelClass,
) -> Result<Vec<String>, AttemptError> {
    let mut guard = pool.write().await;
    let id_refs: Vec<&str> = accounts.iter().map(|a| a.id.as_str()).collect();
    guard.sync_accounts(Provider::Claude, &id_refs);

    let mut ids: Vec<String> = Vec::new();
    if let Some(pref) = preferred {
        if !guard.is_known(Provider::Claude, pref) {
            return Err(AttemptError::RateLimit(RateLimitError {
                status: 429,
                cooldown_ms: 0,
            }));
        }
        ids.push(pref.to_string());
    }
    for id in guard.available_ids(Provider::Claude, class) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    if ids.is_empty() {
        return Err(AttemptError::RateLimit(RateLimitError {
            status: 429,
            cooldown_ms: 0,
        }));
    }
    Ok(ids)
}

fn prepare_claude_attempt_headers(token: &str, client_headers: Option<&HeaderMap>) -> HeaderMap {
    let client_ua = client_headers
        .and_then(|h| h.get("user-agent"))
        .and_then(|v| v.to_str().ok());
    let anthropic_ver = client_headers
        .and_then(|h| h.get("anthropic-version"))
        .and_then(|v| v.to_str().ok());
    let anthropic_beta = client_headers
        .and_then(|h| h.get("anthropic-beta"))
        .and_then(|v| v.to_str().ok());

    let mut headers = build_claude_code_headers(token, anthropic_ver, anthropic_beta, client_ua);
    if let Some(ch) = client_headers {
        for (key, val) in ch {
            let name = key.as_str();
            if (name.starts_with("x-") || name.starts_with("anthropic-"))
                && name != "anthropic-version"
                && name != "anthropic-beta"
                && name != "x-api-key"
            {
                headers.insert(key.clone(), val.clone());
            }
        }
    }
    headers
}

async fn execute_claude_attempt(
    client: &reqwest::Client,
    body: &Value,
    headers: HeaderMap,
    class: ModelClass,
    pool: &Arc<RwLock<AccountPool>>,
    account_id: &str,
) -> Result<ClaudeStreamResult, AttemptError> {
    let res = client
        .post(ANTHROPIC_MESSAGES_URL)
        .headers(headers)
        .json(body)
        .send()
        .await
        .map_err(|e| AttemptError::NonAccount(NonAccountError::new(e.to_string())))?;

    let status = res.status().as_u16();
    if !res.status().is_success() {
        let headers = res.headers().clone();
        let error_text = res.text().await.unwrap_or_default();
        if is_rate_limit_status(status, &error_text) {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs() * 1000 + u64::from(d.subsec_millis()));
            let cooldown_ms = parse_cooldown_ms(&headers, &error_text, now_ms);
            let until = now_ms + cooldown_ms;
            pool.write()
                .await
                .mark_cooldown(Provider::Claude, account_id, class, until);
            return Err(AttemptError::RateLimit(RateLimitError { status, cooldown_ms }));
        }
        if (400..500).contains(&status) {
            return Err(AttemptError::Upstream(UpstreamError::new(
                status,
                format!("Upstream client error ({status}): {error_text}"),
            )));
        }
        return Err(AttemptError::Upstream(UpstreamError::new(
            status,
            format!("Upstream error ({status}): {error_text}"),
        )));
    }

    Ok(ClaudeStreamResult {
        response: res,
        account_id: account_id.to_string(),
    })
}

/// Dispatches a streaming call to Claude upstream using account failover.
///
/// # Errors
/// Returns `AttemptError` if failover exhausts or non-account failure occurs.
pub async fn call_claude_stream(
    params: &CallClaudeParams,
    pool: &Arc<RwLock<AccountPool>>,
    client: &reqwest::Client,
) -> Result<ClaudeStreamResult, AttemptError> {
    let accounts: Vec<ClaudeAccount> = get_claude_accounts()
        .into_iter()
        .filter(|a| !a.claude_ai_oauth.access_token.is_empty())
        .collect();

    if accounts.is_empty() {
        return Err(AttemptError::NonAccount(NonAccountError::new(
            "No configured Claude accounts found",
        )));
    }

    let candidates = resolve_claude_candidates(
        pool,
        &accounts,
        params.preferred_account_id.as_deref(),
        params.model_class,
    )
    .await?;

    let mut last_err = None;
    for account_id in &candidates {
        pool.write().await.stamp_use(Provider::Claude, account_id);
        let Some(account) = accounts.iter().find(|a| a.id == *account_id) else {
            continue;
        };

        let Ok(Some(token)) = get_claude_token(Some(account), Some(client)).await else {
            last_err = Some(AttemptError::NonAccount(NonAccountError::new(
                "Failed to acquire valid Claude access token",
            )));
            continue;
        };

        let mut request_body = params.body.clone();
        let session_id = get_deterministic_session_id(&request_body);
        stamp_claude_metadata(&mut request_body, &session_id);
        inject_claude_code_cache_control(&mut request_body, get_cache_ttl());

        let headers = prepare_claude_attempt_headers(&token, params.client_headers.as_ref());

        match execute_claude_attempt(client, &request_body, headers, params.model_class, pool, account_id).await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if err.is_deterministic_client_error() || err.is_non_account_failure() {
                    return Err(err);
                }
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        AttemptError::RateLimit(RateLimitError {
            status: 429,
            cooldown_ms: 0,
        })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64url_encode() {
        assert_eq!(base64url_encode(b""), "");
        assert_eq!(base64url_encode(b"f"), "Zg");
        assert_eq!(base64url_encode(b"fo"), "Zm8");
        assert_eq!(base64url_encode(b"foo"), "Zm9v");
        assert_eq!(base64url_encode(b"foob"), "Zm9vYg");
        assert_eq!(base64url_encode(b"fooba"), "Zm9vYmE");
        assert_eq!(base64url_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_claude_login_url_structure() {
        let url = get_claude_login_url(9005);
        assert!(url.starts_with("https://claude.com/cai/oauth/authorize?"));
        assert!(url.contains("client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e"));
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A9005%2Fcallback"));
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state="));
    }

    #[test]
    fn test_save_claude_token_data_persists() {
        let acc_id = "test-claude-acc";
        save_claude_token_data(acc_id, "access_123", "refresh_456", 1_234_567_890).unwrap();
        let accounts = get_claude_accounts();
        let saved = accounts.iter().find(|a| a.id == acc_id);
        assert!(saved.is_some());
        let saved = saved.unwrap();
        assert_eq!(saved.claude_ai_oauth.access_token, "access_123");
        assert_eq!(saved.claude_ai_oauth.refresh_token.as_deref(), Some("refresh_456"));
    }

    #[test]
    fn test_map_anthropic_model() {
        assert_eq!(
            map_anthropic_model("claude-3-5-sonnet-20241022-native"),
            "claude-3-5-sonnet-20241022"
        );
        assert_eq!(map_anthropic_model("claude-fable-something"), "claude-fable-5");
        assert_eq!(map_anthropic_model("claude-opus-5"), "claude-opus-5");
        assert_eq!(map_anthropic_model("claude-haiku"), "claude-haiku-4-5-20251001");
        assert_eq!(map_anthropic_model("claude-sonnet-5"), "claude-sonnet-5");
        assert_eq!(map_anthropic_model("unknown"), "claude-sonnet-4-6");
    }

    #[test]
    fn test_normalize_anthropic_effort() {
        assert_eq!(normalize_anthropic_effort(Some("minimal")), Some("low".to_string()));
        assert_eq!(normalize_anthropic_effort(Some("high")), Some("high".to_string()));
        assert_eq!(normalize_anthropic_effort(Some("NONE")), None);
        assert_eq!(normalize_anthropic_effort(None), None);
    }

    #[test]
    fn test_apply_thinking_constraints() {
        let mut body = serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        });
        apply_thinking_constraints(&mut body);
        assert_eq!(body["temperature"], 1);
        assert!(body.get("top_p").is_none());
        assert!(body.get("top_k").is_none());
    }

    #[test]
    fn test_with_claude_code_identity() {
        let sys = serde_json::json!("Be friendly");
        let blocks = with_claude_code_identity(Some(&sys));
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["text"], CLAUDE_CODE_IDENTITY);
        assert_eq!(blocks[1]["text"], "Be friendly");

        // Idempotent if identity already present
        let blocks_again = with_claude_code_identity(Some(&serde_json::Value::Array(blocks.clone())));
        assert_eq!(blocks_again.len(), 2);
    }

    #[test]
    fn test_inject_claude_code_cache_control() {
        let mut body = serde_json::json!({
            "system": [{ "type": "text", "text": "system prompt" }],
            "messages": [
                { "role": "user", "content": "hello" }
            ]
        });
        inject_claude_code_cache_control(&mut body, CacheTtl::FiveMinutes);
        assert!(body["system"][0].get("cache_control").is_some());
        assert!(body["messages"][0]["content"][0].get("cache_control").is_some());
    }

    #[test]
    fn test_build_claude_code_headers() {
        let headers = build_claude_code_headers("token_123", None, None, None);
        assert_eq!(headers.get("authorization").unwrap(), "Bearer token_123");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(
            headers
                .get("user-agent")
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("claude-cli/")
        );
        assert!(
            headers
                .get("anthropic-beta")
                .unwrap()
                .to_str()
                .unwrap()
                .contains("claude-code-20250219")
        );
    }

    #[tokio::test]
    async fn test_call_claude_stream_fails_with_unknown_preferred_account() {
        let pool = Arc::new(RwLock::new(AccountPool::default()));
        let client = reqwest::Client::new();
        let params = CallClaudeParams {
            body: serde_json::json!({
                "model": "claude-sonnet-5",
                "messages": [{ "role": "user", "content": "hi" }]
            }),
            model_class: ModelClass::Claude,
            preferred_account_id: Some("non-existent-account-id".to_string()),
            client_headers: None,
        };

        let res = call_claude_stream(&params, &pool, &client).await;
        assert!(res.is_err());
    }
}
