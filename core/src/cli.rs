use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::Query;
use axum::response::Html;
use axum::routing::get;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::app::{AppState, create_app};
use crate::config::get_port;
use crate::services::anthropic::{exchange_claude_code, get_claude_login_url};
use crate::services::auth::{exchange_google_code, get_login_url};
use crate::services::codex::{exchange_codex_code, get_codex_login_url, store_codex_verifier};
use crate::services::credentials::{
    clear_credentials, get_claude_accounts, get_codex_accounts, get_gemini_accounts, remove_claude_account,
    remove_codex_account, remove_gemini_account,
};

#[derive(Debug, Parser)]
#[command(name = "polyport", version, about = "Polyport AI proxy server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Commands {
    Start(StartArgs),
    Login {
        provider: String,
    },
    Logout {
        provider: String,
        account_id: Option<String>,
    },
    Status,
}

#[derive(Debug, clap::Args, PartialEq, Eq)]
pub struct StartArgs {
    #[arg(short, long)]
    pub port: Option<u16>,
    #[arg(long = "proxy-only", default_value_t = false)]
    pub proxy_only: bool,
    #[arg(long)]
    pub sandbox: Option<String>,
    #[arg(long = "cache-ttl")]
    pub cache_ttl: Option<String>,
}

#[must_use]
pub fn parse_cli_args(argv: &[String]) -> Cli {
    if argv.is_empty() {
        return Cli { command: None };
    }
    match Cli::try_parse_from(std::iter::once("polyport").chain(argv.iter().map(String::as_str))) {
        Ok(cli) => cli,
        Err(_) => Cli { command: None },
    }
}

#[must_use]
pub fn validate_cache_ttl(v: &str) -> bool {
    matches!(v.to_ascii_lowercase().as_str(), "5m" | "1h" | "off")
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| (d.as_secs() * 1000 + u64::from(d.subsec_millis())).cast_signed())
}

fn base64url_nopad(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    let mut i = 0;
    while i < input.len() {
        let b0 = input[i];
        let b1 = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] } else { 0 };
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        let rem = input.len() - i;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if rem > 1 {
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        }
        if rem > 2 {
            out.push(ALPHABET[(n & 63) as usize] as char);
        }
        i += 3;
    }
    out
}

fn random_hex(len: usize) -> String {
    let mut s = String::new();
    while s.len() < len {
        s.push_str(&uuid::Uuid::new_v4().simple().to_string());
    }
    s.chars().take(len).collect()
}

fn code_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64url_nopad(&digest)
}

fn open_browser(url: &str) {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Ok(mut child) = result {
        let _ = child.wait();
    }
}

#[derive(Debug, Deserialize)]
struct CallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

#[must_use]
pub fn render_status() -> String {
    let gemini = get_gemini_accounts();
    let claude = get_claude_accounts();
    let codex = get_codex_accounts();
    let port = get_port();
    let sandbox = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .display()
        .to_string();
    let cache_bytes = file_size(".cache/completions-cache.json");
    let history_bytes = file_size(".cache/history-cache.json");
    let list = |ids: Vec<String>| {
        if ids.is_empty() {
            "    (none)".to_string()
        } else {
            ids.join("\n")
        }
    };
    let gemini_lines = list(gemini.iter().map(|a| format!("    {}: valid", a.id)).collect());
    let claude_lines = list(
        claude
            .iter()
            .map(|a| {
                let state = match a.claude_ai_oauth.expires_at {
                    Some(exp) if exp < now_ms() => "expired",
                    _ => "valid",
                };
                format!("    {}: {state}", a.id)
            })
            .collect(),
    );
    let codex_lines = list(
        codex
            .iter()
            .map(|a| {
                let state = match a.expires_at {
                    Some(exp) if exp < now_ms() => "expired",
                    _ => "valid",
                };
                format!("    {}: {state}", a.id)
            })
            .collect(),
    );
    #[allow(clippy::cast_precision_loss)]
    let cache_kb = cache_bytes as f64 / 1024.0;
    #[allow(clippy::cast_precision_loss)]
    let history_kb = history_bytes as f64 / 1024.0;
    format!(
        "Polyport status\n  Gemini accounts:\n{gemini_lines}\n  Claude accounts:\n{claude_lines}\n  Codex accounts:\n{codex_lines}\n  Port:         {port}\n  Sandbox:      {sandbox}\n  Cache:        {cache_kb:.1} KB (history {history_kb:.1} KB)"
    )
}

fn file_size(p: &str) -> u64 {
    std::fs::metadata(p).map_or(0, |m| m.len())
}

/// # Errors
/// Returns error message string for invalid provider or account combos.
pub fn run_logout(provider: &str, account_id: Option<&str>) -> Result<String, String> {
    if !matches!(provider, "gemini" | "claude" | "codex" | "all") {
        return Err("Usage: polyport logout <gemini|claude|codex|all> [accountId]".to_string());
    }
    if account_id.is_some() && provider == "all" {
        return Err("Cannot combine \"all\" with an account id.".to_string());
    }
    if let Some(id) = account_id {
        let res = match provider {
            "gemini" => remove_gemini_account(id),
            "claude" => remove_claude_account(id),
            "codex" => remove_codex_account(id),
            _ => return Err("Usage: polyport logout <gemini|claude|codex> <accountId>".to_string()),
        };
        res.map_err(|e| e.to_string())?;
        return Ok(format!("Logged out {provider} account: {id}"));
    }
    clear_credentials(provider).map_err(|e| e.to_string())?;
    Ok(format!("Logged out: {provider}"))
}

async fn run_login(provider: &str) -> anyhow::Result<()> {
    let provider = provider.to_ascii_lowercase();
    if !matches!(provider.as_str(), "gemini" | "claude" | "codex") {
        anyhow::bail!("Usage: polyport login <gemini|claude|codex>");
    }
    let port: u16 = if provider == "codex" { 1455 } else { get_port() };
    let (callback_path, auth_url) = match provider.as_str() {
        "gemini" => ("/oauth2callback".to_string(), get_login_url()),
        "claude" => ("/callback".to_string(), get_claude_login_url(port)),
        "codex" => {
            let verifier = random_hex(64);
            let state = random_hex(43);
            let challenge = code_challenge_s256(&verifier);
            store_codex_verifier(&state, &verifier);
            let url = get_codex_login_url(port, &challenge, &state);
            ("/auth/callback".to_string(), url)
        }
        _ => unreachable!(),
    };
    println!("Open this URL to authorize:\n{auth_url}\n");
    open_browser(&auth_url);
    println!("Waiting for OAuth callback on http://localhost:{port}{callback_path}");

    let (tx, rx) = oneshot::channel::<Result<String, String>>();
    let tx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(tx)));
    let provider_clone = provider.clone();
    let app = axum::Router::new().route(
        &callback_path,
        get(move |Query(q): Query<CallbackQuery>| {
            let tx = tx.clone();
            let provider_clone = provider_clone.clone();
            async move {
                let result: Result<String, String> = async {
                    if let Some(err) = q.error {
                        return Err(err);
                    }
                    let code = q.code.ok_or_else(|| "Authorization code missing".to_string())?;
                    match provider_clone.as_str() {
                        "gemini" => {
                            let id = exchange_google_code(&code, None).await.map_err(|e| e.to_string())?;
                            Ok(format!("Logged in as {id}"))
                        }
                        "claude" => {
                            exchange_claude_code(&code, q.state.as_deref(), None)
                                .await
                                .map_err(|e| e.to_string())?;
                            Ok("Claude login successful".to_string())
                        }
                        "codex" => {
                            let verifier = crate::services::codex::get_stored_codex_verifier(q.state.as_deref())
                                .ok_or_else(|| "Session expired".to_string())?;
                            exchange_codex_code(&code, &verifier, port, None)
                                .await
                                .map_err(|e| e.to_string())?;
                            Ok("Codex login successful".to_string())
                        }
                        _ => Err("Unknown provider".to_string()),
                    }
                }
                .await;
                let html = match &result {
                    Ok(msg) => format!("<h1>Login successful</h1><p>{msg}. You can close this tab.</p>"),
                    Err(err) => format!("<h1>Login failed</h1><p>{err}</p>"),
                };
                if let Some(tx) = tx.lock().await.take() {
                    let _ = tx.send(result);
                }
                Html(html)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = rx.await;
        })
        .await?;
    Ok(())
}

/// # Errors
/// Returns error for invalid port, invalid cache TTL, or sandbox setup failure.
#[allow(clippy::unused_async)]
pub async fn prepare_start(args: &StartArgs) -> anyhow::Result<u16> {
    if let Some(port) = args.port {
        if port == 0 {
            anyhow::bail!("Invalid port: {port}");
        }
        // SAFETY: single-threaded startup before serving; no concurrent readers yet.
        unsafe {
            std::env::set_var("PORT", port.to_string());
        }
    }
    if args.proxy_only {
        // SAFETY: startup configuration before server tasks spawn.
        unsafe {
            std::env::set_var("POLYPORT_PROXY_ONLY", "1");
        }
    }
    if let Some(ttl) = &args.cache_ttl {
        if !validate_cache_ttl(ttl) {
            anyhow::bail!("--cache-ttl must be one of 5m|1h|off (got: {ttl})");
        }
        // SAFETY: startup configuration before server tasks spawn.
        unsafe {
            std::env::set_var("POLYPORT_CACHE_TTL", ttl);
        }
    }
    let sandbox_path = if let Some(s) = &args.sandbox {
        PathBuf::from(s)
    } else {
        PathBuf::from("sandbox")
    };
    std::fs::create_dir_all(&sandbox_path)?;
    std::env::set_current_dir(&sandbox_path)?;

    if get_gemini_accounts().is_empty() {
        println!("[Auth] Gemini credentials missing. Run: polyport login gemini");
    }
    if get_claude_accounts().is_empty() {
        println!("[Claude Auth] Claude credentials missing. Run: polyport login claude");
    }
    if get_codex_accounts().is_empty() {
        println!("[Codex Auth] Codex credentials missing. Run: polyport login codex");
    }

    Ok(get_port())
}

/// # Errors
/// Returns error if the TCP bind or serve fails.
pub async fn serve_router(app: axum::Router, port: u16) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_start(args: &StartArgs) -> anyhow::Result<()> {
    let port = prepare_start(args).await?;
    let state = AppState::default();
    let app = create_app(state);
    serve_router(app, port).await
}

/// # Errors
/// Returns error if server, login exchange, or logout operation fails.
pub async fn run_cli(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        None => {
            run_start(&StartArgs {
                port: None,
                proxy_only: false,
                sandbox: None,
                cache_ttl: None,
            })
            .await
        }
        Some(Commands::Start(args)) => run_start(&args).await,
        Some(Commands::Login { provider }) => run_login(&provider).await,
        Some(Commands::Logout { provider, account_id }) => match run_logout(&provider, account_id.as_deref()) {
            Ok(msg) => {
                println!("{msg}");
                Ok(())
            }
            Err(msg) => {
                eprintln!("{msg}");
                anyhow::bail!("{msg}");
            }
        },
        Some(Commands::Status) => {
            println!("{}", render_status());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Restore {
        cwd: Option<std::path::PathBuf>,
        port: Option<String>,
        proxy_only: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.port {
                    Some(v) => std::env::set_var("PORT", v),
                    None => std::env::remove_var("PORT"),
                }
                match &self.proxy_only {
                    Some(v) => std::env::set_var("POLYPORT_PROXY_ONLY", v),
                    None => std::env::remove_var("POLYPORT_PROXY_ONLY"),
                }
            }
            if let Some(cwd) = &self.cwd {
                let _ = std::env::set_current_dir(cwd);
            }
        }
    }

    #[test]
    fn test_empty_argv_defaults_to_start() {
        let cli = parse_cli_args(&[]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn test_login_provider_parses() {
        let argv = vec!["login".to_string(), "gemini".to_string()];
        let cli = parse_cli_args(&argv);
        assert_eq!(
            cli.command,
            Some(Commands::Login {
                provider: "gemini".to_string()
            })
        );
    }

    #[test]
    fn test_start_port_parses() {
        let raw = vec!["start".to_string(), "--port".to_string(), "9005".to_string()];
        let cli = parse_cli_args(&raw);
        match cli.command {
            Some(Commands::Start(opts)) => assert_eq!(opts.port, Some(9005)),
            _ => panic!("expected start"),
        }
    }

    #[test]
    fn test_cache_ttl_validation() {
        assert!(validate_cache_ttl("5m"));
        assert!(validate_cache_ttl("1h"));
        assert!(validate_cache_ttl("off"));
        assert!(!validate_cache_ttl("10m"));
    }

    #[test]
    fn test_logout_rejects_all_with_account() {
        let err = run_logout("all", Some("x")).unwrap_err();
        assert!(err.contains("Cannot combine"));
    }

    #[test]
    fn test_logout_rejects_unknown_provider() {
        assert!(run_logout("bogus", None).is_err());
    }

    #[test]
    fn test_code_challenge_is_url_safe() {
        let challenge = code_challenge_s256("test-verifier-123");
        assert!(!challenge.contains('='));
        assert!(!challenge.contains('+'));
        assert!(!challenge.contains('/'));
        assert_eq!(challenge.len(), 43);
    }

    #[test]
    fn test_render_status_contains_sections() {
        let status = render_status();
        assert!(status.contains("Polyport status"));
        assert!(status.contains("Gemini accounts"));
        assert!(status.contains("Claude accounts"));
        assert!(status.contains("Codex accounts"));
    }

    #[tokio::test]
    async fn test_prepare_start_sets_port_env() {
        let saved_cwd = std::env::current_dir().ok();
        let saved_port = std::env::var("PORT").ok();
        let saved_proxy_only = std::env::var("POLYPORT_PROXY_ONLY").ok();
        let _restore = Restore {
            cwd: saved_cwd,
            port: saved_port,
            proxy_only: saved_proxy_only,
        };
        unsafe {
            std::env::remove_var("PORT");
        }
        let port = prepare_start(&StartArgs {
            port: Some(19191),
            proxy_only: true,
            sandbox: Some("/tmp/polyport-plan-w1".into()),
            cache_ttl: None,
        })
        .await
        .unwrap();
        assert_eq!(port, 19191);
        assert_eq!(std::env::var("PORT").unwrap(), "19191");
    }
}
