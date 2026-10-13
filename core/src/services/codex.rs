use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::services::account_pool::{AccountPool, AttemptError};
use crate::services::credentials::{
    CodexAccount, CodexTokens, decode_chatgpt_account_id, decode_oidc_account_id, get_codex_accounts,
    upsert_codex_account,
};
use crate::types::{ModelClass, Provider};
use crate::utils::rate_limit::{
    NonAccountError, RateLimitError, UpstreamError, is_rate_limit_status, parse_cooldown_ms,
};

pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

static PENDING_CODEX_VERIFIERS: Mutex<VecDeque<(String, String)>> = Mutex::new(VecDeque::new());

pub fn store_codex_verifier(state: &str, verifier: &str) {
    if let Ok(mut lock) = PENDING_CODEX_VERIFIERS.lock() {
        if lock.len() >= 100 {
            lock.pop_front();
        }
        lock.push_back((state.to_string(), verifier.to_string()));
    }
}

#[must_use]
pub fn get_stored_codex_verifier(state: Option<&str>) -> Option<String> {
    let mut lock = PENDING_CODEX_VERIFIERS.lock().ok()?;
    if let Some(s) = state {
        let pos_opt = lock.iter().position(|(k, _)| k == s);
        if let Some(pos) = pos_opt {
            return lock.remove(pos).map(|(_, v)| v);
        }
    }
    lock.pop_back().map(|(_, v)| v)
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
pub fn has_codex_token() -> bool {
    !get_codex_accounts().is_empty()
}

#[must_use]
pub fn get_codex_login_url(port: u16, code_challenge: &str, state: &str) -> String {
    let redirect_uri = url_encode(&format!("http://localhost:{port}/auth/callback"));
    let scopes = url_encode("openid profile email offline_access api.connectors.read api.connectors.invoke");

    format!(
        "https://auth.openai.com/oauth/authorize?response_type=code&client_id={CODEX_CLIENT_ID}&redirect_uri={redirect_uri}&scope={scopes}&code_challenge={code_challenge}&code_challenge_method=S256&id_token_add_organizations=true&codex_cli_simplified_flow=true&state={state}&originator=codex_cli_rs"
    )
}

/// # Errors
/// Fails if credentials store cannot be written.
pub fn save_codex_token_data(
    account_id: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    id_token: Option<&str>,
    openai_api_key: Option<&str>,
    expires_in: u64,
) -> anyhow::Result<()> {
    let existing = get_codex_accounts().into_iter().find(|a| a.id == account_id);

    let final_refresh = refresh_token
        .map(ToString::to_string)
        .or_else(|| existing.as_ref().and_then(|a| a.tokens.refresh_token.clone()));
    let final_id_token = id_token
        .map(ToString::to_string)
        .or_else(|| existing.as_ref().and_then(|a| a.tokens.id_token.clone()));

    let real_account_id =
        decode_chatgpt_account_id(final_id_token.as_deref()).unwrap_or_else(|| account_id.to_string());

    let expires_in_ms = expires_in.cast_signed() * 1000;
    let expires_at = now_epoch_ms() + expires_in_ms;

    let auth_method = if openai_api_key.is_some() { "api_key" } else { "chatgpt" };

    upsert_codex_account(&CodexAccount {
        id: account_id.to_string(),
        tokens: CodexTokens {
            access_token: Some(access_token.to_string()),
            refresh_token: final_refresh,
            id_token: final_id_token,
            account_id: Some(real_account_id),
        },
        auth_method: Some(auth_method.to_string()),
        openai_api_key: openai_api_key.map(ToString::to_string),
        expires_at: Some(expires_at),
    })
}

#[derive(Debug, Deserialize)]
struct CodexTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
}

/// # Errors
/// Fails if token exchange fails or response is invalid.
pub async fn exchange_codex_code(
    code: &str,
    code_verifier: &str,
    port: u16,
    client: Option<&reqwest::Client>,
) -> anyhow::Result<()> {
    let redirect_uri = format!("http://localhost:{port}/auth/callback");
    let default_client = reqwest::Client::new();
    let http = client.unwrap_or(&default_client);

    let body = format!(
        "client_id={CODEX_CLIENT_ID}&code={code}&grant_type=authorization_code&redirect_uri={}&code_verifier={code_verifier}",
        url_encode(&redirect_uri)
    );

    let res = http
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !res.status().is_success() {
        let err_text = res.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Codex token exchange failed: {err_text}"));
    }

    let data: CodexTokenResponse = res.json().await?;
    let account_id =
        decode_oidc_account_id(data.id_token.as_deref()).unwrap_or_else(|| format!("codex-{}", now_epoch_ms()));

    save_codex_token_data(
        &account_id,
        &data.access_token,
        data.refresh_token.as_deref(),
        data.id_token.as_deref(),
        None,
        data.expires_in.unwrap_or(3600),
    )
}

async fn try_refresh_codex_token(
    http: &reqwest::Client,
    acc_id: &str,
    refresh_token: &str,
    api_key: Option<&str>,
) -> Option<String> {
    let body = format!("client_id={CODEX_CLIENT_ID}&refresh_token={refresh_token}&grant_type=refresh_token");

    let res = http
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .ok()?;

    if !res.status().is_success() {
        return None;
    }

    let data: CodexTokenResponse = res.json().await.ok()?;
    let new_refresh = data.refresh_token.as_deref().unwrap_or(refresh_token);
    let _ = save_codex_token_data(
        acc_id,
        &data.access_token,
        Some(new_refresh),
        data.id_token.as_deref(),
        api_key,
        data.expires_in.unwrap_or(3600),
    );

    Some(data.access_token)
}

/// Retrieves an active Codex token, refreshing proactively if within 5 minutes of expiration.
///
/// # Errors
/// Returns an error if refreshing fails and no existing token is valid.
pub async fn get_codex_token(
    account: Option<&CodexAccount>,
    client: Option<&reqwest::Client>,
) -> anyhow::Result<Option<String>> {
    let default_accounts = get_codex_accounts();
    let acc = account.or_else(|| default_accounts.first());

    let Some(acc) = acc else {
        return Ok(None);
    };

    if let (Some("api_key"), Some(key)) = (acc.auth_method.as_deref(), &acc.openai_api_key) {
        return Ok(Some(key.clone()));
    }

    let Some(ref access_token) = acc.tokens.access_token else {
        return Ok(None);
    };

    let now = now_epoch_ms();
    let expires_at = acc.expires_at.unwrap_or(0);
    let is_expiring = expires_at > 0 && (expires_at - now) < 5 * 60 * 1000;

    let refresh_token = match (is_expiring, &acc.tokens.refresh_token) {
        (true, Some(rt)) => Some(rt.as_str()),
        _ => None,
    };

    if let Some(rt) = refresh_token {
        let default_client = reqwest::Client::new();
        let http = client.unwrap_or(&default_client);
        if let Some(new_token) = try_refresh_codex_token(http, &acc.id, rt, acc.openai_api_key.as_deref()).await {
            return Ok(Some(new_token));
        }
    }

    Ok(Some(access_token.clone()))
}

#[derive(Debug, Clone)]
pub struct CallCodexParams {
    pub body: Value,
    pub model_class: ModelClass,
    pub preferred_account_id: Option<String>,
    pub conversation_id: Option<String>,
}

#[derive(Debug)]
pub struct CodexStreamResult {
    pub response: reqwest::Response,
    pub account_id: String,
}

async fn resolve_codex_candidates(
    pool: &Arc<RwLock<AccountPool>>,
    accounts: &[CodexAccount],
    preferred: Option<&str>,
    class: ModelClass,
) -> Result<Vec<String>, AttemptError> {
    let mut guard = pool.write().await;
    let id_refs: Vec<&str> = accounts.iter().map(|a| a.id.as_str()).collect();
    guard.sync_accounts(Provider::Codex, &id_refs);

    let mut ids: Vec<String> = Vec::new();
    if let Some(pref) = preferred {
        if !guard.is_known(Provider::Codex, pref) {
            return Err(AttemptError::RateLimit(RateLimitError {
                status: 429,
                cooldown_ms: 0,
            }));
        }
        ids.push(pref.to_string());
    }
    for id in guard.available_ids(Provider::Codex, class) {
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

fn prepare_codex_headers(token: &str, account: &CodexAccount, conversation_id: Option<&str>) -> HeaderMap {
    let mut map = HeaderMap::new();
    let insert = |h: &mut HeaderMap, k: &'static str, v: &str| {
        if let Ok(val) = HeaderValue::from_str(v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };

    insert(&mut map, "content-type", "application/json");
    insert(&mut map, "accept", "text/event-stream");
    insert(&mut map, "originator", "codex_cli_rs");
    insert(&mut map, "authorization", &format!("Bearer {token}"));

    if let (Some("chatgpt"), Some(act_id)) = (account.auth_method.as_deref(), &account.tokens.account_id) {
        insert(&mut map, "chatgpt-account-id", act_id);
    }

    if let Some(cid) = conversation_id {
        insert(&mut map, "session-id", cid);
        insert(&mut map, "thread-id", cid);
    }

    map
}

async fn execute_codex_attempt(
    client: &reqwest::Client,
    body: &Value,
    headers: HeaderMap,
    class: ModelClass,
    pool: &Arc<RwLock<AccountPool>>,
    account_id: &str,
) -> Result<CodexStreamResult, AttemptError> {
    let res = client
        .post(CODEX_RESPONSES_URL)
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
                .mark_cooldown(Provider::Codex, account_id, class, until);
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

    Ok(CodexStreamResult {
        response: res,
        account_id: account_id.to_string(),
    })
}

/// Dispatches a streaming call to Codex Responses API upstream using account failover.
///
/// # Errors
/// Returns `AttemptError` if failover exhausts or non-account failure occurs.
pub async fn call_codex_stream(
    params: &CallCodexParams,
    pool: &Arc<RwLock<AccountPool>>,
    client: &reqwest::Client,
) -> Result<CodexStreamResult, AttemptError> {
    let accounts: Vec<CodexAccount> = get_codex_accounts()
        .into_iter()
        .filter(|a| a.tokens.access_token.is_some() || a.openai_api_key.is_some())
        .collect();

    if accounts.is_empty() {
        return Err(AttemptError::NonAccount(NonAccountError::new(
            "No configured Codex accounts found",
        )));
    }

    let candidates = resolve_codex_candidates(
        pool,
        &accounts,
        params.preferred_account_id.as_deref(),
        params.model_class,
    )
    .await?;

    let mut last_err = None;
    for account_id in &candidates {
        pool.write().await.stamp_use(Provider::Codex, account_id);
        let Some(account) = accounts.iter().find(|a| a.id == *account_id) else {
            continue;
        };

        let Ok(Some(token)) = get_codex_token(Some(account), Some(client)).await else {
            last_err = Some(AttemptError::NonAccount(NonAccountError::new(
                "Failed to acquire valid Codex access token",
            )));
            continue;
        };

        let headers = prepare_codex_headers(&token, account, params.conversation_id.as_deref());

        match execute_codex_attempt(client, &params.body, headers, params.model_class, pool, account_id).await {
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
    fn test_codex_login_url_contains_expected_params() {
        let url = get_codex_login_url(9005, "challenge123", "state456");
        assert!(url.starts_with("https://auth.openai.com/oauth/authorize?"));
        assert!(url.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A9005%2Fauth%2Fcallback"));
        assert!(url.contains("code_challenge=challenge123"));
        assert!(url.contains("state=state456"));
        assert!(url.contains("originator=codex_cli_rs"));
    }

    #[test]
    fn test_save_codex_token_data_persists_round_trip() {
        let acc_id = "test-codex-user";
        save_codex_token_data(acc_id, "access_token_123", Some("refresh_token_456"), None, None, 3600).unwrap();

        let accounts = get_codex_accounts();
        let saved = accounts.iter().find(|a| a.id == acc_id);
        assert!(saved.is_some());
        let saved = saved.unwrap();
        assert_eq!(saved.tokens.access_token.as_deref(), Some("access_token_123"));
        assert_eq!(saved.tokens.refresh_token.as_deref(), Some("refresh_token_456"));
    }

    #[tokio::test]
    async fn test_call_codex_stream_fails_with_unknown_preferred_account() {
        let pool = Arc::new(RwLock::new(AccountPool::default()));
        let client = reqwest::Client::new();
        let params = CallCodexParams {
            body: serde_json::json!({
                "model": "gpt-5.5",
                "input": []
            }),
            model_class: ModelClass::Codex,
            preferred_account_id: Some("non-existent-account-id".to_string()),
            conversation_id: None,
        };

        let res = call_codex_stream(&params, &pool, &client).await;
        assert!(res.is_err());
    }
}
