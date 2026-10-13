use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenInfo {
    pub access_token: String,
    pub token_type: Option<String>,
    pub refresh_token: Option<String>,
    pub expiry: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiAccount {
    pub id: String,
    pub token: TokenInfo,
    pub auth_method: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaudeOauth {
    #[serde(rename = "accessToken")]
    pub access_token: String,
    #[serde(rename = "refreshToken", skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(rename = "expiresAt", skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaudeAccount {
    pub id: String,
    #[serde(rename = "claudeAiOauth")]
    pub claude_ai_oauth: ClaudeOauth,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodexTokens {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub account_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodexAccount {
    pub id: String,
    pub tokens: CodexTokens,
    pub auth_method: Option<String>,
    #[serde(rename = "OPENAI_API_KEY", skip_serializing_if = "Option::is_none")]
    pub openai_api_key: Option<String>,
    #[serde(rename = "expiresAt", skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Credentials {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gemini: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claude: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codex: Option<Value>,
}

#[must_use]
pub fn compute_credential_path(
    platform: &str,
    home: &Path,
    xdg_config_home: Option<&Path>,
    root_cwd: &Path,
    local_override_exists: bool,
) -> PathBuf {
    if local_override_exists {
        return root_cwd.join(".polyport.json");
    }
    if platform == "win32" {
        return home.join(".polyport").join("credential.json");
    }
    let base = match xdg_config_home {
        Some(dir) => dir.to_path_buf(),
        None => home.join(".config"),
    };
    base.join("polyport").join("credential.json")
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

#[must_use]
pub fn get_credential_path() -> PathBuf {
    let root_cwd = std::env::var("POLYPORT_ROOT_CWD").unwrap_or_else(|_| ".".to_owned());
    let root_cwd = PathBuf::from(root_cwd);
    let local_override = root_cwd.join(".polyport.json");
    compute_credential_path(
        if std::env::consts::OS == "windows" {
            "win32"
        } else {
            std::env::consts::OS
        },
        &home_dir(),
        std::env::var_os("XDG_CONFIG_HOME").as_deref().map(Path::new),
        &root_cwd,
        local_override.exists(),
    )
}

/// # Errors
/// Fails when the file cannot be read or holds invalid JSON.
pub fn read_credentials() -> anyhow::Result<Credentials> {
    let path = get_credential_path();
    if !path.exists() {
        return Ok(Credentials::default());
    }
    let content = std::fs::read_to_string(&path)?;
    if content.trim().is_empty() {
        return Ok(Credentials::default());
    }
    serde_json::from_str(&content).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse credentials file at {}: {e}. Please fix or delete the file.",
            path.display()
        )
    })
}

#[cfg(unix)]
fn write_private_tmp(tmp: &Path, content: &str) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(tmp)?;
    std::io::Write::write_all(&mut file, content.as_bytes())
}

#[cfg(not(unix))]
fn write_private_tmp(tmp: &Path, content: &str) -> std::io::Result<()> {
    std::fs::write(tmp, content)
}

/// # Errors
/// Fails when the directory, temp file, or final rename cannot be written.
pub fn write_credentials(creds: &Credentials) -> anyhow::Result<()> {
    let path = get_credential_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_json::to_string_pretty(creds)?;
    let tmp = path.with_extension("json.tmp");
    write_private_tmp(&tmp, &content)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn clear_credentials(provider: &str) -> anyhow::Result<()> {
    if provider == "all" {
        return write_credentials(&Credentials::default());
    }
    let mut creds = read_credentials()?;
    match provider {
        "gemini" => creds.gemini = None,
        "claude" => creds.claude = None,
        "codex" => creds.codex = None,
        _ => {}
    }
    write_credentials(&creds)
}

fn accounts_in(section: Option<&Value>) -> Vec<Value> {
    match section.and_then(|s| s.get("accounts")) {
        Some(Value::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

#[must_use]
pub fn get_gemini_accounts() -> Vec<GeminiAccount> {
    let creds = read_credentials().unwrap_or_default();
    accounts_in(creds.gemini.as_ref())
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect()
}

#[must_use]
pub fn get_claude_accounts() -> Vec<ClaudeAccount> {
    claude_accounts_in(&read_credentials().unwrap_or_default())
}

#[must_use]
pub fn claude_accounts_in(creds: &Credentials) -> Vec<ClaudeAccount> {
    accounts_in(creds.claude.as_ref())
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect()
}

#[must_use]
pub fn get_codex_accounts() -> Vec<CodexAccount> {
    let creds = read_credentials().unwrap_or_default();
    accounts_in(creds.codex.as_ref())
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect()
}

fn accounts_section(accounts: &[Value]) -> Value {
    serde_json::json!({ "accounts": accounts })
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn upsert_gemini_account(account: &GeminiAccount) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.gemini.as_ref());
    let value = serde_json::to_value(account)?;
    if let Some(pos) = list.iter().position(|v| v.get("id") == value.get("id")) {
        list[pos] = value;
    } else {
        list.push(value);
    }
    creds.gemini = Some(accounts_section(&list));
    write_credentials(&creds)
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn upsert_claude_account(account: &ClaudeAccount) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.claude.as_ref());
    let value = serde_json::to_value(account)?;
    if let Some(pos) = list.iter().position(|v| v.get("id") == value.get("id")) {
        list[pos] = value;
    } else {
        list.push(value);
    }
    creds.claude = Some(accounts_section(&list));
    write_credentials(&creds)
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn upsert_codex_account(account: &CodexAccount) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.codex.as_ref());
    let value = serde_json::to_value(account)?;
    if let Some(pos) = list.iter().position(|v| v.get("id") == value.get("id")) {
        list[pos] = value;
    } else {
        list.push(value);
    }
    creds.codex = Some(accounts_section(&list));
    write_credentials(&creds)
}

fn remove_from(list: &mut Vec<Value>, id: &str) {
    list.retain(|v| v.get("id").and_then(Value::as_str) != Some(id));
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn remove_gemini_account(id: &str) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.gemini.as_ref());
    remove_from(&mut list, id);
    creds.gemini = Some(accounts_section(&list));
    write_credentials(&creds)
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn remove_claude_account(id: &str) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.claude.as_ref());
    remove_from(&mut list, id);
    creds.claude = Some(accounts_section(&list));
    write_credentials(&creds)
}

/// # Errors
/// Fails when the store cannot be read or written.
pub fn remove_codex_account(id: &str) -> anyhow::Result<()> {
    let mut creds = read_credentials()?;
    let mut list = accounts_in(creds.codex.as_ref());
    remove_from(&mut list, id);
    creds.codex = Some(accounts_section(&list));
    write_credentials(&creds)
}

fn b64url_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

fn b64url_decode(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    if text.is_empty() || text.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let bytes = text.as_bytes();
    let full = bytes.len() / 4 * 4;
    for quad in bytes[..full].chunks(4) {
        let n = (u32::from(b64url_value(quad[0])?) << 18)
            | (u32::from(b64url_value(quad[1])?) << 12)
            | (u32::from(b64url_value(quad[2])?) << 6)
            | u32::from(b64url_value(quad[3])?);
        out.extend_from_slice(&n.to_be_bytes()[1..4]);
    }
    match &bytes[full..] {
        [] => {}
        [a, b] => {
            let n = (u32::from(b64url_value(*a)?) << 18) | (u32::from(b64url_value(*b)?) << 12);
            out.extend_from_slice(&n.to_be_bytes()[1..2]);
        }
        [a, b, c] => {
            let n = (u32::from(b64url_value(*a)?) << 18)
                | (u32::from(b64url_value(*b)?) << 12)
                | (u32::from(b64url_value(*c)?) << 6);
            out.extend_from_slice(&n.to_be_bytes()[1..3]);
        }
        _ => return None,
    }
    Some(out)
}

#[must_use]
pub fn decode_oidc_account_id(id_token: Option<&str>) -> Option<String> {
    let payload_part = id_token?.split('.').nth(1)?;
    let bytes = b64url_decode(payload_part)?;
    let payload: Value = serde_json::from_slice(&bytes).ok()?;
    payload
        .get("email")
        .and_then(Value::as_str)
        .or_else(|| payload.get("sub").and_then(Value::as_str))
        .map(str::to_owned)
}

#[must_use]
pub fn decode_chatgpt_account_id(id_token: Option<&str>) -> Option<String> {
    let payload_part = id_token?.split('.').nth(1)?;
    let bytes = b64url_decode(payload_part)?;
    let payload: Value = serde_json::from_slice(&bytes).ok()?;
    payload
        .get("https://api.openai.com/auth")
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// # Errors
/// Fails when neither email nor uuid is present.
pub fn derive_claude_account_id(email: Option<&str>, account_uuid: Option<&str>) -> anyhow::Result<String> {
    if let Some(email) = email {
        return Ok(email.to_owned());
    }
    if let Some(uuid) = account_uuid {
        return Ok(format!("claude-{uuid}"));
    }
    anyhow::bail!("OAuth token-exchange response is missing account identity (email_address / uuid).")
}

#[cfg(test)]
mod test_helpers {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    pub struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
        _guard: MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        pub fn capture(keys: &[&str]) -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let saved = keys.iter().map(|k| ((*k).to_owned(), std::env::var(k).ok())).collect();
            Self { saved, _guard: guard }
        }

        pub fn set(key: &str, value: &str) {
            // SAFETY: every env mutation in this test binary holds ENV_LOCK.
            unsafe {
                std::env::set_var(key, value);
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in std::mem::take(&mut self.saved) {
                // SAFETY: same lock as set.
                unsafe {
                    match value {
                        Some(val) => std::env::set_var(&key, val),
                        None => std::env::remove_var(&key),
                    }
                }
            }
        }
    }

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn fresh(prefix: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!("{prefix}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        pub fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    pub fn b64url_encode(input: &str) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::new();
        for chunk in input.as_bytes().chunks(3) {
            let b0 = u32::from(chunk[0]);
            let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
            let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
            let group = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[((group >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((group >> 12) & 63) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((group >> 6) & 63) as usize] as char);
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(group & 63) as usize] as char);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::{EnvGuard, TempDir, b64url_encode};
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn local_override_wins() {
        assert_eq!(
            compute_credential_path("linux", Path::new("/h"), None, Path::new("/work"), true),
            PathBuf::from("/work/.polyport.json")
        );
    }

    #[test]
    fn linux_uses_xdg_config_home() {
        assert_eq!(
            compute_credential_path(
                "linux",
                Path::new("/h"),
                Some(Path::new("/xdg")),
                Path::new("/work"),
                false
            ),
            PathBuf::from("/xdg/polyport/credential.json")
        );
    }

    #[test]
    fn linux_falls_back_to_home_dot_config() {
        assert_eq!(
            compute_credential_path("linux", Path::new("/h"), None, Path::new("/work"), false),
            PathBuf::from("/h/.config/polyport/credential.json")
        );
    }

    #[test]
    fn win32_uses_home_dot_polyport() {
        assert_eq!(
            compute_credential_path("win32", Path::new("/h"), None, Path::new("/work"), false),
            PathBuf::from("/h/.polyport/credential.json")
        );
    }

    #[test]
    fn helpers_stay_linked() {
        let _guard = EnvGuard::capture(&[]);
        let tmp = TempDir::fresh("cred-link");
        EnvGuard::set("POLYPORT_TEST_LINK", tmp.path().to_str().unwrap());
        assert_ne!(b64url_encode("link"), "");
    }

    fn isolated_env(prefix: &str) -> (TempDir, EnvGuard) {
        let tmp = TempDir::fresh(prefix);
        let guard = EnvGuard::capture(&["HOME", "XDG_CONFIG_HOME", "POLYPORT_ROOT_CWD"]);
        EnvGuard::set("HOME", tmp.path().to_str().unwrap());
        EnvGuard::set("XDG_CONFIG_HOME", &tmp.path().join("config").to_string_lossy());
        EnvGuard::set("POLYPORT_ROOT_CWD", tmp.path().to_str().unwrap());
        (tmp, guard)
    }

    #[test]
    fn write_then_read_round_trips() {
        let (_tmp, _guard) = isolated_env("cred-rw");
        write_credentials(&Credentials {
            gemini: Some(serde_json::json!({"token": {"access_token": "a"}})),
            claude: None,
            codex: None,
        })
        .unwrap();
        assert_eq!(
            read_credentials().unwrap(),
            Credentials {
                gemini: Some(serde_json::json!({"token": {"access_token": "a"}})),
                claude: None,
                codex: None,
            }
        );
    }

    #[test]
    fn read_missing_file_returns_empty() {
        let (_tmp, _guard) = isolated_env("cred-missing");
        assert_eq!(read_credentials().unwrap(), Credentials::default());
    }

    #[test]
    fn clear_one_provider_keeps_the_other() {
        let (_tmp, _guard) = isolated_env("cred-clear-one");
        write_credentials(&Credentials {
            gemini: Some(serde_json::json!({"token": {"access_token": "a"}})),
            claude: Some(serde_json::json!({"claudeAiOauth": {"accessToken": "c"}})),
            codex: None,
        })
        .unwrap();
        clear_credentials("gemini").unwrap();
        assert_eq!(
            read_credentials().unwrap(),
            Credentials {
                gemini: None,
                claude: Some(serde_json::json!({"claudeAiOauth": {"accessToken": "c"}})),
                codex: None,
            }
        );
    }

    #[test]
    fn clear_all_empties_the_store() {
        let (_tmp, _guard) = isolated_env("cred-clear-all");
        write_credentials(&Credentials {
            gemini: Some(serde_json::json!({"token": {"access_token": "a"}})),
            claude: None,
            codex: None,
        })
        .unwrap();
        clear_credentials("all").unwrap();
        assert_eq!(read_credentials().unwrap(), Credentials::default());
    }

    #[test]
    fn credential_path_honors_local_override() {
        let (tmp, _guard) = isolated_env("cred-override");
        std::fs::write(tmp.path().join(".polyport.json"), "{}").unwrap();
        assert_eq!(get_credential_path(), tmp.path().join(".polyport.json"));
    }

    #[test]
    fn read_malformed_file_errors() {
        let (_tmp, _guard) = isolated_env("cred-malformed");
        let path = get_credential_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "not-valid-json").unwrap();
        let err = read_credentials().unwrap_err().to_string();
        assert!(err.contains("Failed to parse credentials file"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn legacy_single_object_shapes_are_ignored() {
        let (_tmp, _guard) = isolated_env("cred-legacy");
        write_credentials(&Credentials {
            gemini: Some(serde_json::json!({"token": {"access_token": "a"}, "auth_method": "consumer"})),
            claude: Some(serde_json::json!({"claudeAiOauth": {"accessToken": "c"}})),
            codex: Some(serde_json::json!({"tokens": {"access_token": "k"}})),
        })
        .unwrap();
        assert_eq!(get_gemini_accounts(), Vec::new());
        assert_eq!(get_claude_accounts(), Vec::new());
        assert_eq!(get_codex_accounts(), Vec::new());
    }

    #[test]
    fn empty_store_gives_empty_arrays() {
        let (_tmp, _guard) = isolated_env("cred-empty");
        assert_eq!(get_gemini_accounts(), Vec::new());
        assert_eq!(get_claude_accounts(), Vec::new());
        assert_eq!(get_codex_accounts(), Vec::new());
    }

    #[test]
    fn upsert_appends_and_updates_by_id() {
        let (_tmp, _guard) = isolated_env("cred-upsert");
        upsert_gemini_account(&GeminiAccount {
            id: "x@g.com".to_owned(),
            token: TokenInfo {
                access_token: "t1".to_owned(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        })
        .unwrap();
        upsert_gemini_account(&GeminiAccount {
            id: "y@g.com".to_owned(),
            token: TokenInfo {
                access_token: "t2".to_owned(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        })
        .unwrap();
        let ids: Vec<String> = get_gemini_accounts().iter().map(|a| a.id.clone()).collect();
        assert_eq!(ids, vec!["x@g.com", "y@g.com"]);
        upsert_gemini_account(&GeminiAccount {
            id: "x@g.com".to_owned(),
            token: TokenInfo {
                access_token: "t1-new".to_owned(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        })
        .unwrap();
        let accounts = get_gemini_accounts();
        assert_eq!(accounts.len(), 2);
        assert_eq!(
            accounts.iter().find(|a| a.id == "x@g.com").unwrap().token.access_token,
            "t1-new"
        );
    }

    #[test]
    fn upsert_over_legacy_shape_writes_array_only() {
        let (_tmp, _guard) = isolated_env("cred-upsert-legacy");
        write_credentials(&Credentials {
            gemini: Some(serde_json::json!({"token": {"access_token": "legacy"}, "auth_method": "consumer"})),
            claude: None,
            codex: None,
        })
        .unwrap();
        upsert_gemini_account(&GeminiAccount {
            id: "new@g.com".to_owned(),
            token: TokenInfo {
                access_token: "fresh".to_owned(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        })
        .unwrap();
        let ids: Vec<String> = get_gemini_accounts().iter().map(|a| a.id.clone()).collect();
        assert_eq!(ids, vec!["new@g.com"]);
    }

    #[test]
    fn remove_drops_one_account_by_id() {
        let (_tmp, _guard) = isolated_env("cred-remove");
        for (id, token) in [("a", "aa"), ("b", "bb")] {
            upsert_claude_account(&ClaudeAccount {
                id: id.to_owned(),
                claude_ai_oauth: ClaudeOauth {
                    access_token: token.to_owned(),
                    refresh_token: None,
                    expires_at: None,
                },
            })
            .unwrap();
        }
        remove_claude_account("a").unwrap();
        let ids: Vec<String> = get_claude_accounts().iter().map(|a| a.id.clone()).collect();
        assert_eq!(ids, vec!["b"]);
    }

    #[test]
    fn codex_upsert_and_remove_round_trip() {
        let (_tmp, _guard) = isolated_env("cred-codex");
        upsert_codex_account(&CodexAccount {
            id: "dev".to_owned(),
            tokens: CodexTokens {
                access_token: Some("k".to_owned()),
                refresh_token: None,
                id_token: None,
                account_id: Some("acc-1".to_owned()),
            },
            auth_method: Some("chatgpt".to_owned()),
            openai_api_key: None,
            expires_at: None,
        })
        .unwrap();
        assert_eq!(get_codex_accounts().len(), 1);
        remove_codex_account("dev").unwrap();
        assert_eq!(get_codex_accounts(), Vec::new());
    }

    #[test]
    fn gemini_remove_drops_by_id() {
        let (_tmp, _guard) = isolated_env("cred-gemini-remove");
        upsert_gemini_account(&GeminiAccount {
            id: "gone@g.com".to_owned(),
            token: TokenInfo {
                access_token: "t".to_owned(),
                token_type: None,
                refresh_token: None,
                expiry: None,
            },
            auth_method: None,
        })
        .unwrap();
        remove_gemini_account("gone@g.com").unwrap();
        assert_eq!(get_gemini_accounts(), Vec::new());
    }

    #[test]
    fn oidc_reads_email_then_sub() {
        let payload = b64url_encode(r#"{"email":"me@g.com","sub":"123"}"#);
        assert_eq!(
            decode_oidc_account_id(Some(&format!("x.{payload}.y"))).as_deref(),
            Some("me@g.com")
        );
        let sub = b64url_encode(r#"{"sub":"999"}"#);
        assert_eq!(
            decode_oidc_account_id(Some(&format!("x.{sub}.y"))).as_deref(),
            Some("999")
        );
        let chatgpt = b64url_encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acc-xyz-123"}}"#);
        assert_eq!(
            decode_chatgpt_account_id(Some(&format!("x.{chatgpt}.y"))).as_deref(),
            Some("acc-xyz-123")
        );
        assert_eq!(decode_chatgpt_account_id(None), None);
        assert_eq!(decode_oidc_account_id(None), None);
        assert_eq!(decode_oidc_account_id(Some("not-a-jwt")), None);
    }

    #[test]
    fn claude_id_prefers_email_then_prefixed_uuid() {
        assert_eq!(
            derive_claude_account_id(Some("me@example.com"), Some("u1")).unwrap(),
            "me@example.com"
        );
        assert_eq!(
            derive_claude_account_id(None, Some("abc-123")).unwrap(),
            "claude-abc-123"
        );
        assert!(
            derive_claude_account_id(None, None)
                .unwrap_err()
                .to_string()
                .contains("account identity")
        );
    }
}
