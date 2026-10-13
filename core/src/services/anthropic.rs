use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::config::get_port;
use crate::services::credentials::{
    ClaudeAccount, ClaudeOauth, derive_claude_account_id, get_claude_accounts, upsert_claude_account,
};

pub const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const CLAUDE_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

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
}
