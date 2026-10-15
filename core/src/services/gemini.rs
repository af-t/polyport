use std::sync::Arc;
use tokio::sync::RwLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS, SUPPORTED_MODELS, max_gemini_output_tokens};
use crate::services::account_pool::{AccountPool, AttemptError};
use crate::services::auth::get_or_refresh_google_token;
use crate::services::credentials::{GeminiAccount, get_gemini_accounts};
use crate::services::gcp::get_gcp_project_id;
use crate::types::{ModelClass, Provider};
use crate::utils::rate_limit::{
    NonAccountError, RateLimitError, UpstreamError, is_rate_limit_status, parse_cooldown_ms,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    pub include_thoughts: bool,
    pub thinking_budget: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[must_use]
pub fn build_generation_config(body: &Value, resolved_model: &str, is_thinking_requested: bool) -> GenerationConfig {
    let model_lower = resolved_model.to_ascii_lowercase();
    let model_info = SUPPORTED_MODELS
        .iter()
        .find(|m| m.id == resolved_model || m.id == model_lower);

    let is_anthropic_owned = model_info.map_or_else(
        || model_lower.contains("claude"),
        |info| info.owned_by == "anthropic" || model_lower.contains("claude"),
    );

    let default_max_tokens = if is_anthropic_owned {
        DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS
    } else {
        max_gemini_output_tokens(resolved_model)
    };

    let requested_max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(default_max_tokens);

    let mut max_output_tokens = requested_max_tokens.max(256);
    let temperature = body.get("temperature").and_then(Value::as_f64).or(Some(0.7));
    let top_p = body.get("top_p").and_then(Value::as_f64);
    let top_k = body.get("top_k").and_then(Value::as_u64);

    let stop_sequences = body.get("stop").and_then(|s| {
        if let Some(arr) = s.as_array() {
            Some(
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect::<Vec<_>>(),
            )
        } else {
            s.as_str().map(|str_val| vec![str_val.to_string()])
        }
    });

    let mut thinking_config = None;
    if is_thinking_requested {
        let mut budget: i64 = body
            .get("thinking")
            .and_then(|t| t.get("budget_tokens"))
            .or_else(|| body.get("reasoning").and_then(|r| r.get("max_tokens")))
            .or_else(|| body.get("thinking_budget"))
            .and_then(Value::as_i64)
            .unwrap_or(-1);

        if budget == -1 {
            let effort = body
                .get("reasoning_effort")
                .or_else(|| body.get("reasoning").and_then(|r| r.get("effort")))
                .and_then(Value::as_str);

            if let Some(eff) = effort {
                let eff_lower = eff.to_ascii_lowercase();
                budget = match eff_lower.as_str() {
                    "low" | "minimal" => 1024,
                    "medium" => 4096,
                    "high" => 8192,
                    "xhigh" => 16384,
                    "max" => 32768,
                    _ => -1,
                };
            }
        }

        if budget == -1 && model_lower.contains("claude") {
            budget = 1024;
        }

        if model_lower.contains("claude")
            && budget > 0
            && let Ok(b_u32) = u32::try_from(budget)
            && max_output_tokens <= b_u32
        {
            max_output_tokens = b_u32 + 1024;
        }

        thinking_config = Some(ThinkingConfig {
            include_thoughts: true,
            thinking_budget: budget,
        });
    }

    GenerationConfig {
        max_output_tokens: Some(max_output_tokens),
        temperature,
        top_p,
        top_k,
        stop_sequences,
        thinking_config,
    }
}

#[derive(Debug, Clone)]
pub struct CallGeminiParams {
    pub contents: Vec<Value>,
    pub system_instruction: Option<Value>,
    pub generation_config: GenerationConfig,
    pub tools: Option<Vec<Value>>,
    pub tool_config: Option<Value>,
    pub upstream_model: String,
    pub model_class: ModelClass,
    pub preferred_account_id: Option<String>,
}

pub struct GeminiStreamResult {
    pub response: reqwest::Response,
    pub account_id: String,
}

#[must_use]
pub fn build_upstream_payload(params: &CallGeminiParams, project_id: &str) -> Value {
    let mut request_obj = serde_json::json!({
        "contents": params.contents,
        "generationConfig": params.generation_config,
    });

    if let Some(sys) = &params.system_instruction {
        request_obj["systemInstruction"] = sys.clone();
    }
    if let Some(tools) = &params.tools {
        request_obj["tools"] = serde_json::json!(tools);
    }
    if let Some(tc) = &params.tool_config {
        request_obj["toolConfig"] = tc.clone();
    }

    let request_id = format!("api-{}", uuid::Uuid::new_v4());

    serde_json::json!({
        "project": project_id,
        "requestId": request_id,
        "model": params.upstream_model,
        "userAgent": "antigravity",
        "requestType": "chat",
        "request": request_obj
    })
}

async fn execute_gemini_attempt(
    client: &reqwest::Client,
    account: Option<&GeminiAccount>,
    upstream_payload: &Value,
    class: ModelClass,
    pool: &Arc<RwLock<AccountPool>>,
    account_id: &str,
) -> Result<GeminiStreamResult, AttemptError> {
    let token = get_or_refresh_google_token(account, Some(client))
        .await
        .map_err(|e| AttemptError::Other(anyhow::anyhow!("Auth failed: {e}")))?;

    tracing::info!(
        "[Proxy] Routing to Gemini pool (model: \"{}\", account: {account_id})",
        upstream_payload
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );

    let res = client
        .post("https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse")
        .header("Authorization", format!("Bearer {token}"))
        .header("User-Agent", "antigravity/cli/1.3.5 linux/arm64")
        .header("Content-Type", "application/json")
        .json(upstream_payload)
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
                .mark_cooldown(Provider::Gemini, account_id, class, until);
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

    Ok(GeminiStreamResult {
        response: res,
        account_id: account_id.to_string(),
    })
}

/// Dispatches a streaming call to Gemini upstream using account failover.
///
/// # Errors
/// Returns `AttemptError` if failover exhausts or non-account failure occurs.
pub async fn call_gemini_stream(
    params: &CallGeminiParams,
    pool: &Arc<RwLock<AccountPool>>,
    client: &reqwest::Client,
) -> Result<GeminiStreamResult, AttemptError> {
    let accounts: Vec<GeminiAccount> = get_gemini_accounts()
        .into_iter()
        .filter(|a| !a.token.access_token.is_empty())
        .collect();

    if accounts.is_empty() {
        return Err(AttemptError::NonAccount(NonAccountError::new(
            "No configured Gemini accounts found",
        )));
    }

    let project_id = get_gcp_project_id();
    let upstream_payload = build_upstream_payload(params, &project_id);

    let preferred = params.preferred_account_id.as_deref();
    let class = params.model_class;

    let candidates: Vec<String> = {
        let mut guard = pool.write().await;
        let id_refs: Vec<&str> = accounts.iter().map(|a| a.id.as_str()).collect();
        guard.sync_accounts(Provider::Gemini, &id_refs);

        let mut ids: Vec<String> = Vec::new();
        if let Some(pref) = preferred {
            if !guard.is_known(Provider::Gemini, pref) {
                return Err(AttemptError::RateLimit(RateLimitError {
                    status: 429,
                    cooldown_ms: 0,
                }));
            }
            ids.push(pref.to_string());
        }
        for id in guard.available_ids(Provider::Gemini, class) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    };

    if candidates.is_empty() {
        return Err(AttemptError::RateLimit(RateLimitError {
            status: 429,
            cooldown_ms: 0,
        }));
    }

    let mut last_err = None;
    for account_id in &candidates {
        pool.write().await.stamp_use(Provider::Gemini, account_id);
        let Some(account) = accounts.iter().find(|a| a.id == *account_id) else {
            continue;
        };

        match execute_gemini_attempt(client, Some(account), &upstream_payload, class, pool, account_id).await {
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
    fn test_get_gcp_project_id_fallback() {
        let pid = crate::services::gcp::get_gcp_project_id();
        assert_ne!(pid, "");
    }

    #[test]
    fn test_build_generation_config_default_and_clamps() {
        let body = serde_json::json!({
            "temperature": 0.5,
            "max_tokens": 1000,
            "stop": ["END"]
        });
        let config = build_generation_config(&body, "gemini-3.5-flash", false);
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_output_tokens, Some(1000));
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
        assert!(config.thinking_config.is_none());
    }

    #[test]
    fn test_build_generation_config_thinking_effort() {
        let body = serde_json::json!({
            "reasoning_effort": "high"
        });
        let config = build_generation_config(&body, "gemini-3.8-flash-tiered", true);
        assert!(config.thinking_config.is_some());
        let tc = config.thinking_config.unwrap();
        assert!(tc.include_thoughts);
        assert_eq!(tc.thinking_budget, 8192);
    }

    #[test]
    fn test_build_generation_config_dynamic_thinking_when_no_budget() {
        let body = serde_json::json!({});
        let config = build_generation_config(&body, "gemini-3.8-flash-tiered", true);
        assert!(config.thinking_config.is_some());
        let tc = config.thinking_config.unwrap();
        assert!(tc.include_thoughts);
        assert_eq!(tc.thinking_budget, -1);
    }

    #[test]
    fn test_build_upstream_payload_structure() {
        let params = CallGeminiParams {
            contents: vec![serde_json::json!({ "role": "user", "parts": [{ "text": "hello" }] })],
            system_instruction: None,
            generation_config: GenerationConfig::default(),
            tools: None,
            tool_config: None,
            upstream_model: "gemini-3.5-flash".to_string(),
            model_class: crate::types::ModelClass::GeminiNative,
            preferred_account_id: None,
        };
        let payload = build_upstream_payload(&params, "test-project-123");
        assert_eq!(payload.get("project").and_then(Value::as_str), Some("test-project-123"));
        assert_eq!(payload.get("model").and_then(Value::as_str), Some("gemini-3.5-flash"));
        assert_eq!(payload.get("userAgent").and_then(Value::as_str), Some("antigravity"));
        assert!(payload.get("request").is_some());
    }

    #[tokio::test]
    async fn test_call_gemini_stream_fails_with_unknown_preferred_account() {
        let params = CallGeminiParams {
            contents: vec![],
            system_instruction: None,
            generation_config: GenerationConfig::default(),
            tools: None,
            tool_config: None,
            upstream_model: "gemini-3.5-flash".to_string(),
            model_class: crate::types::ModelClass::GeminiNative,
            preferred_account_id: Some("nonexistent-account".to_string()),
        };
        let pool = std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::services::account_pool::AccountPool::default(),
        ));
        let client = reqwest::Client::new();
        let res = call_gemini_stream(&params, &pool, &client).await;
        assert!(res.is_err());
    }
}
