use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::config::{CLIENT_ID, CLIENT_SECRET, get_port};
use crate::services::credentials::{
    GeminiAccount, TokenInfo, decode_oidc_account_id, get_gemini_accounts, upsert_gemini_account,
};

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("OAuth token not found. Please log in first by navigating to: {0}")]
    TokenNotFound(String),
    #[error("No refresh token found in token data")]
    NoRefreshToken,
    #[error("Token exchange failed: {0}")]
    ExchangeFailed(String),
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Credential store error: {0}")]
    Credential(String),
}

#[derive(Debug, Deserialize)]
struct TokenExchangeResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    id_token: Option<String>,
}

fn url_encode(input: &str) -> String {
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

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() * 1000 + u64::from(d.subsec_millis()))
}

fn format_iso8601(system_time: SystemTime) -> String {
    let dur = system_time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let total_secs = dur.as_secs();
    let millis = dur.subsec_millis();

    let days = (total_secs / 86400).cast_signed();
    let rem_secs = total_secs % 86400;
    let hours = rem_secs / 3600;
    let minutes = (rem_secs % 3600) / 60;
    let seconds = rem_secs % 60;

    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}.{millis:03}Z")
}

fn parse_iso8601_epoch_ms(input_str: &str) -> Option<u64> {
    if let Ok(num) = input_str.parse::<u64>() {
        return Some(if num < 10_000_000_000 { num * 1000 } else { num });
    }

    let trimmed = input_str.trim().trim_end_matches('Z');
    let (date_part, time_part) = trimmed.split_once('T')?;
    let mut date_iter = date_part.split('-');
    let parsed_year: i64 = date_iter.next()?.parse().ok()?;
    let parsed_month: i64 = date_iter.next()?.parse().ok()?;
    let parsed_day: i64 = date_iter.next()?.parse().ok()?;

    let (time_str, millis_str) = match time_part.split_once('.') {
        Some((time_val, ms_val)) => (time_val, ms_val),
        None => (time_part, "0"),
    };
    let mut time_iter = time_str.split(':');
    let hour_val: u64 = time_iter.next()?.parse().ok()?;
    let minute_val: u64 = time_iter.next()?.parse().ok()?;
    let second_val: u64 = time_iter.next()?.parse().ok()?;
    let millis_val: u64 = millis_str.chars().take(3).collect::<String>().parse().ok().unwrap_or(0);

    let year_shifted = if parsed_month <= 2 {
        parsed_year - 1
    } else {
        parsed_year
    };
    let era = year_shifted / 400;
    let yoe = year_shifted - era * 400;
    let doy =
        (153 * (if parsed_month > 2 {
            parsed_month - 3
        } else {
            parsed_month + 9
        }) + 2)
            / 5
            + parsed_day
            - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;

    let total_secs = days.cast_unsigned() * 86400 + hour_val * 3600 + minute_val * 60 + second_val;
    Some(total_secs * 1000 + millis_val)
}

#[must_use]
pub fn get_login_url() -> String {
    let redirect_uri = format!("http://localhost:{}/oauth2callback", get_port());
    let scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "openid",
        "email",
        "profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ]
    .join(" ");

    format!(
        "https://accounts.google.com/o/oauth2/v2/auth?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
        url_encode(CLIENT_ID),
        url_encode(&redirect_uri),
        url_encode(&scopes),
    )
}

/// Saves OAuth access and refresh token data into the credential store.
///
/// # Errors
/// Returns `AuthError::Credential` if saving to the credential file fails.
pub fn save_token_data(
    account_id: &str,
    access_token: &str,
    refresh_token: Option<&str>,
    expires_in_secs: u64,
) -> Result<(), AuthError> {
    let existing = get_gemini_accounts().into_iter().find(|a| a.id == account_id);
    let final_refresh = refresh_token
        .map(String::from)
        .or_else(|| existing.and_then(|a| a.token.refresh_token));

    let now = SystemTime::now();
    let expiry_time = now + Duration::from_secs(expires_in_secs);
    let expiry_iso = format_iso8601(expiry_time);

    let account = GeminiAccount {
        id: account_id.to_string(),
        token: TokenInfo {
            access_token: access_token.to_string(),
            token_type: Some("Bearer".to_string()),
            refresh_token: final_refresh,
            expiry: Some(expiry_iso),
        },
        auth_method: Some("consumer".to_string()),
    };

    upsert_gemini_account(&account).map_err(|e| AuthError::Credential(e.to_string()))?;
    Ok(())
}

/// Exchanges Google authorization code for token data.
///
/// # Errors
/// Returns `AuthError` on network failure or if the Google API returns an error status.
pub async fn exchange_google_code(code: &str, client: Option<&reqwest::Client>) -> Result<String, AuthError> {
    let default_client = reqwest::Client::new();
    let http = client.unwrap_or(&default_client);

    let redirect_uri = format!("http://localhost:{}/oauth2callback", get_port());
    let params = [
        ("client_id", CLIENT_ID),
        ("client_secret", CLIENT_SECRET),
        ("code", code),
        ("grant_type", "authorization_code"),
        ("redirect_uri", &redirect_uri),
    ];
    let body = params
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    let res = http
        .post("https://oauth2.googleapis.com/token")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !res.status().is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(AuthError::ExchangeFailed(text));
    }

    let data: TokenExchangeResponse = res.json().await?;
    let now_ms = now_epoch_ms();
    let account_id = decode_oidc_account_id(data.id_token.as_deref()).unwrap_or_else(|| format!("gemini-{now_ms}"));

    save_token_data(
        &account_id,
        &data.access_token,
        data.refresh_token.as_deref(),
        data.expires_in,
    )?;

    Ok(account_id)
}

static LAST_REFRESH_MAP: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Reads and refreshes access token for the given account if near expiry (< 5 minutes).
///
/// # Errors
/// Returns `AuthError::TokenNotFound` if account has no access token, or `AuthError` on refresh failure.
pub async fn get_or_refresh_google_token(
    account: Option<&GeminiAccount>,
    client: Option<&reqwest::Client>,
) -> Result<String, AuthError> {
    let stored_accounts = get_gemini_accounts();
    let acc = account.or_else(|| stored_accounts.first());

    let Some(acc) = acc else {
        return Err(AuthError::TokenNotFound(get_login_url()));
    };

    if acc.token.access_token.is_empty() {
        return Err(AuthError::TokenNotFound(get_login_url()));
    }

    let now_ms = now_epoch_ms();
    let expiry_ms = acc
        .token
        .expiry
        .as_deref()
        .and_then(parse_iso8601_epoch_ms)
        .unwrap_or(0);

    // Refresh if expiring in < 5 minutes (300,000 ms)
    if expiry_ms > 0 && expiry_ms.saturating_sub(now_ms) < 300_000 {
        let should_refresh = {
            let mut guard = LAST_REFRESH_MAP
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let map = guard.get_or_insert_with(HashMap::new);
            let last = map.get(&acc.id).copied().unwrap_or(0);
            if now_ms.saturating_sub(last) > 120_000 {
                map.insert(acc.id.clone(), now_ms);
                true
            } else {
                false
            }
        };

        if should_refresh {
            let refresh_token = acc.token.refresh_token.as_deref().ok_or(AuthError::NoRefreshToken)?;

            let default_client = reqwest::Client::new();
            let http = client.unwrap_or(&default_client);

            let params = [
                ("client_id", CLIENT_ID),
                ("client_secret", CLIENT_SECRET),
                ("refresh_token", refresh_token),
                ("grant_type", "refresh_token"),
            ];
            let body = params
                .iter()
                .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
                .collect::<Vec<_>>()
                .join("&");

            let res = http
                .post("https://oauth2.googleapis.com/token")
                .header("Content-Type", "application/x-www-form-urlencoded")
                .body(body)
                .send()
                .await?;

            if !res.status().is_success() {
                let text = res.text().await.unwrap_or_default();
                return Err(AuthError::ExchangeFailed(text));
            }

            let data: TokenExchangeResponse = res.json().await?;
            save_token_data(
                &acc.id,
                &data.access_token,
                data.refresh_token.as_deref(),
                data.expires_in,
            )?;

            return Ok(data.access_token);
        }
    }

    Ok(acc.token.access_token.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_login_url_contains_client_id_and_scopes() {
        let url = get_login_url();
        assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?"));
        assert!(url.contains("client_id="));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=consent"));
        assert!(url.contains("oauth2callback"));
    }

    #[test]
    fn test_save_token_data_persists_account() {
        let (_tmp, _guard) = crate::services::credentials::test_helpers::isolated_env("auth-save-gemini");
        let account_id = "test-auth-save-gemini";
        save_token_data(account_id, "mock-access", Some("mock-refresh"), 3600).unwrap();

        let accounts = crate::services::credentials::get_gemini_accounts();
        let found = accounts.iter().find(|a| a.id == account_id).unwrap();
        assert_eq!(found.token.access_token, "mock-access");
        assert_eq!(found.token.refresh_token.as_deref(), Some("mock-refresh"));
    }

    #[test]
    fn test_get_or_refresh_token_non_expiring_returns_existing() {
        let now_future = SystemTime::now() + Duration::from_secs(3600);
        let account = crate::services::credentials::GeminiAccount {
            id: "fresh-account".to_string(),
            token: crate::services::credentials::TokenInfo {
                access_token: "existing-access".to_string(),
                token_type: Some("Bearer".to_string()),
                refresh_token: Some("refresh-val".to_string()),
                expiry: Some(format_iso8601(now_future)),
            },
            auth_method: Some("consumer".to_string()),
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let token = rt.block_on(get_or_refresh_google_token(Some(&account), None)).unwrap();
        assert_eq!(token, "existing-access");
    }

    #[test]
    fn test_get_or_refresh_token_missing_token_errors() {
        let account = crate::services::credentials::GeminiAccount {
            id: "empty-account".to_string(),
            token: crate::services::credentials::TokenInfo {
                access_token: String::new(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(get_or_refresh_google_token(Some(&account), None));
        assert!(res.is_err());
    }
}
