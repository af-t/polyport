use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS, SUPPORTED_MODELS, max_gemini_output_tokens};

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

        if budget > 0 || model_lower.contains("claude") {
            thinking_config = Some(ThinkingConfig {
                include_thoughts: true,
                thinking_budget: budget,
            });
        }
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
}
