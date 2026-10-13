use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::services::credentials::{
    CodexAccount, CodexTokens, decode_chatgpt_account_id, decode_oidc_account_id, get_codex_accounts,
    upsert_codex_account,
};

pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

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
}
