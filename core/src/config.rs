use serde::{Deserialize, Serialize};

use crate::types::{ModelClass, Provider};

pub const DEFAULT_GEMINI_MAX_OUTPUT_TOKENS: u32 = 65_536;
pub const DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS: u32 = 128_000;

pub const CLIENT_ID: &str = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
pub const CLIENT_SECRET: &str = "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";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeModelTier {
    Opus,
    Sonnet,
    Haiku,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ModelEffort {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: &'static str,
    pub owned_by: &'static str,
    pub upstream_name: &'static str,
    pub model_class: Option<ModelClass>,
    pub claude_tier: Option<ClaudeModelTier>,
    pub upstream_by_effort: Option<&'static [(&'static str, &'static str)]>,
    pub default_effort: Option<&'static str>,
}

pub static SUPPORTED_MODELS: &[ModelInfo] = &[
    // Google/Gemini models
    ModelInfo {
        id: "gemini-3.5-flash-low",
        owned_by: "google",
        upstream_name: "gemini-3.5-flash-low",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gemini-3.1-pro-low",
        owned_by: "google",
        upstream_name: "gemini-3.1-pro-low",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gemini-3.1-pro",
        owned_by: "google",
        upstream_name: "gemini-3.1-pro-low",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: Some(&[
            ("low", "gemini-3.1-pro-low"),
            ("medium", "gemini-pro-agent"),
            ("high", "gemini-pro-agent"),
            ("xhigh", "gemini-pro-agent"),
            ("max", "gemini-pro-agent"),
        ]),
        default_effort: Some("low"),
    },
    ModelInfo {
        id: "gemini-3.5-flash",
        owned_by: "google",
        upstream_name: "gemini-3.5-flash-low",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: Some(&[
            ("low", "gemini-3.5-flash-low"),
            ("medium", "gemini-3-flash-agent"),
            ("high", "gemini-3-flash-agent"),
            ("xhigh", "gemini-3-flash-agent"),
            ("max", "gemini-3-flash-agent"),
        ]),
        default_effort: Some("low"),
    },
    ModelInfo {
        id: "gemini-3.6-flash",
        owned_by: "google",
        upstream_name: "gemini-3.6-flash-medium",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: Some(&[
            ("low", "gemini-3.6-flash-low"),
            ("medium", "gemini-3.6-flash-medium"),
            ("high", "gemini-3.6-flash-high"),
        ]),
        default_effort: Some("medium"),
    },
    ModelInfo {
        id: "gemini-3.8-flash",
        owned_by: "google",
        upstream_name: "gemini-3.8-flash-high",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: Some(&[
            ("low", "gemini-3.8-flash-low"),
            ("medium", "gemini-3.8-flash-medium"),
            ("high", "gemini-3.8-flash-high"),
        ]),
        default_effort: Some("high"),
    },
    ModelInfo {
        id: "gemini-3-flash",
        owned_by: "google",
        upstream_name: "gemini-3-flash",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gemini-3.5-flash-lite",
        owned_by: "google",
        upstream_name: "gemini-3.5-flash-lite",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gemini-3.8-flash-tiered",
        owned_by: "google",
        upstream_name: "gemini-3.8-flash-tiered",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-oss-120b-medium",
        owned_by: "google",
        upstream_name: "gpt-oss-120b-medium",
        model_class: Some(ModelClass::GeminiNative),
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-sonnet-4-6",
        owned_by: "google",
        upstream_name: "claude-sonnet-4-6",
        model_class: Some(ModelClass::Claude),
        claude_tier: Some(ClaudeModelTier::Sonnet),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-4-6-thinking",
        owned_by: "google",
        upstream_name: "claude-opus-4-6-thinking",
        model_class: Some(ModelClass::Claude),
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    // Anthropic Native models
    ModelInfo {
        id: "claude-sonnet-5-native",
        owned_by: "anthropic",
        upstream_name: "claude-sonnet-5",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Sonnet),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-sonnet-4-6-native",
        owned_by: "anthropic",
        upstream_name: "claude-sonnet-4-6",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Sonnet),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-4-7-native",
        owned_by: "anthropic",
        upstream_name: "claude-opus-4-7",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-4-8-native",
        owned_by: "anthropic",
        upstream_name: "claude-opus-4-8",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-5-native",
        owned_by: "anthropic",
        upstream_name: "claude-opus-5",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-haiku-4-5-native",
        owned_by: "anthropic",
        upstream_name: "claude-haiku-4-5-20251001",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Haiku),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-fable-native",
        owned_by: "anthropic",
        upstream_name: "claude-fable-5",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-haiku-4-5",
        owned_by: "anthropic",
        upstream_name: "claude-haiku-4-5-20251001",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Haiku),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-haiku-4-5-20251001",
        owned_by: "anthropic",
        upstream_name: "claude-haiku-4-5-20251001",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Haiku),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-4-7",
        owned_by: "anthropic",
        upstream_name: "claude-opus-4-7",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-4-8",
        owned_by: "anthropic",
        upstream_name: "claude-opus-4-8",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-opus-5",
        owned_by: "anthropic",
        upstream_name: "claude-opus-5",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-fable-5",
        owned_by: "anthropic",
        upstream_name: "claude-fable-5",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-sonnet-5",
        owned_by: "anthropic",
        upstream_name: "claude-sonnet-5",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Sonnet),
        upstream_by_effort: None,
        default_effort: None,
    },
    // Codex models
    ModelInfo {
        id: "gpt-5.6-sol",
        owned_by: "codex",
        upstream_name: "gpt-5.6-sol",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-5.6-terra",
        owned_by: "codex",
        upstream_name: "gpt-5.6-terra",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-5.6-luna",
        owned_by: "codex",
        upstream_name: "gpt-5.6-luna",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-6-luna",
        owned_by: "codex",
        upstream_name: "gpt-6-luna",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-5.5",
        owned_by: "codex",
        upstream_name: "gpt-5.5",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-5.4",
        owned_by: "codex",
        upstream_name: "gpt-5.4",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "gpt-5.4-mini",
        owned_by: "codex",
        upstream_name: "gpt-5.4-mini",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "codex-auto-review",
        owned_by: "codex",
        upstream_name: "codex-auto-review",
        model_class: None,
        claude_tier: None,
        upstream_by_effort: None,
        default_effort: None,
    },
    // Aliases
    ModelInfo {
        id: "claude-3-5-sonnet",
        owned_by: "google",
        upstream_name: "claude-sonnet-4-6",
        model_class: Some(ModelClass::Claude),
        claude_tier: Some(ClaudeModelTier::Sonnet),
        upstream_by_effort: None,
        default_effort: None,
    },
    ModelInfo {
        id: "claude-3-opus",
        owned_by: "anthropic",
        upstream_name: "claude-opus-4-8",
        model_class: None,
        claude_tier: Some(ClaudeModelTier::Opus),
        upstream_by_effort: None,
        default_effort: None,
    },
];

#[must_use]
pub fn find_model_info(model: &str) -> Option<&'static ModelInfo> {
    if let Some(info) = SUPPORTED_MODELS.iter().find(|m| m.id == model) {
        return Some(info);
    }
    let lower = model.to_ascii_lowercase();
    SUPPORTED_MODELS.iter().find(|m| m.id == lower)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
#[error("model: {model} is not supported")]
pub struct ModelNotFoundError {
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub provider: Provider,
    pub model_class: ModelClass,
    pub upstream_model: String,
}

/// Resolves a requested model name to a routing decision.
///
/// # Errors
/// Returns `ModelNotFoundError` if `requested_model` is not in `SUPPORTED_MODELS`.
pub fn resolve_route(requested_model: &str, effort: Option<&str>) -> Result<RouteDecision, ModelNotFoundError> {
    if requested_model.is_empty() {
        return Err(ModelNotFoundError { model: String::new() });
    }

    let info = find_model_info(requested_model).ok_or_else(|| ModelNotFoundError {
        model: requested_model.to_string(),
    })?;

    let provider = match info.owned_by {
        "anthropic" => Provider::Claude,
        "codex" => Provider::Codex,
        _ => Provider::Gemini,
    };

    let model_class = match provider {
        Provider::Gemini => info.model_class.unwrap_or(ModelClass::GeminiNative),
        Provider::Claude => ModelClass::Claude,
        Provider::Codex => ModelClass::Codex,
    };

    let upstream_model = effort
        .and_then(|eff| {
            info.upstream_by_effort
                .and_then(|list| list.iter().find(|(k, _)| *k == eff).map(|(_, v)| *v))
        })
        .or_else(|| {
            info.default_effort.and_then(|eff| {
                info.upstream_by_effort
                    .and_then(|list| list.iter().find(|(k, _)| *k == eff).map(|(_, v)| *v))
            })
        })
        .unwrap_or(info.upstream_name);

    Ok(RouteDecision {
        provider,
        model_class,
        upstream_model: upstream_model.to_string(),
    })
}

#[must_use]
pub fn model_supports_thinking(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("gemini-3.1-pro") || lower.contains("gemini-3.6-flash") {
        return true;
    }
    !lower.contains("low")
        && (lower.contains("thinking")
            || lower.contains("agent")
            || lower.contains("opus")
            || lower.contains("sonnet")
            || lower.contains("fable")
            || lower.contains("flash"))
}

#[must_use]
pub fn max_gemini_output_tokens(model: &str) -> u32 {
    let lower = model.to_ascii_lowercase();
    if lower.contains("claude") {
        64_000
    } else if lower.contains("gemini-3.1-pro") || lower.contains("gemini-pro-agent") {
        65_535
    } else {
        DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
    }
}

#[must_use]
pub fn get_port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|val| val.parse::<u16>().ok())
        .unwrap_or(9000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_route_exact_and_case_insensitive() {
        let route = resolve_route("gemini-3.5-flash-low", None).unwrap();
        assert_eq!(route.provider, crate::types::Provider::Gemini);
        assert_eq!(route.model_class, crate::types::ModelClass::GeminiNative);
        assert_eq!(route.upstream_model, "gemini-3.5-flash-low");

        let route_upper = resolve_route("GEMINI-3.5-FLASH-LOW", None).unwrap();
        assert_eq!(route_upper, route);
    }

    #[test]
    fn test_resolve_route_effort_and_default_effort() {
        let route_default = resolve_route("gemini-3.1-pro", None).unwrap();
        assert_eq!(route_default.upstream_model, "gemini-3.1-pro-low");

        let route_med = resolve_route("gemini-3.1-pro", Some("medium")).unwrap();
        assert_eq!(route_med.upstream_model, "gemini-pro-agent");
    }

    #[test]
    fn test_resolve_route_aliases() {
        let sonnet = resolve_route("claude-3-5-sonnet", None).unwrap();
        assert_eq!(sonnet.provider, crate::types::Provider::Gemini);
        assert_eq!(sonnet.model_class, crate::types::ModelClass::Claude);
        assert_eq!(sonnet.upstream_model, "claude-sonnet-4-6");

        let opus = resolve_route("claude-3-opus", None).unwrap();
        assert_eq!(opus.provider, crate::types::Provider::Claude);
        assert_eq!(opus.model_class, crate::types::ModelClass::Claude);
        assert_eq!(opus.upstream_model, "claude-opus-4-8");
    }

    #[test]
    fn test_resolve_route_unknown_model_returns_error() {
        let err = resolve_route("nonexistent-model", None).unwrap_err();
        assert_eq!(err.model, "nonexistent-model");
        assert_eq!(err.to_string(), "model: nonexistent-model is not supported");

        assert!(resolve_route("", None).is_err());
    }

    #[test]
    fn test_model_supports_thinking() {
        assert!(model_supports_thinking("gemini-3.8-flash"));
        assert!(model_supports_thinking("gemini-3.8-flash-high"));
        assert!(model_supports_thinking("gemini-3.8-flash-tiered"));
        assert!(model_supports_thinking("claude-opus-4-6-thinking"));
        assert!(!model_supports_thinking("gemini-3.8-flash-low"));
        assert!(!model_supports_thinking("gpt-oss-120b-medium"));
    }

    #[test]
    fn test_max_gemini_output_tokens() {
        assert_eq!(max_gemini_output_tokens("claude-sonnet-4-6"), 64_000);
        assert_eq!(max_gemini_output_tokens("gemini-3.1-pro"), 65_535);
        assert_eq!(max_gemini_output_tokens("gemini-pro-agent"), 65_535);
        assert_eq!(max_gemini_output_tokens("gemini-3.5-flash"), 65_536);
        assert_eq!(max_gemini_output_tokens("gemini-3.8-flash-high"), 65_536);
    }

    #[test]
    fn test_resolve_route_gemini_38_flash_default_high() {
        let route = resolve_route("gemini-3.8-flash", None).unwrap();
        assert_eq!(route.provider, crate::types::Provider::Gemini);
        assert_eq!(route.model_class, crate::types::ModelClass::GeminiNative);
        assert_eq!(route.upstream_model, "gemini-3.8-flash-high");
    }

    #[test]
    fn test_resolve_route_gemini_38_flash_effort_tiers() {
        let low = resolve_route("gemini-3.8-flash", Some("low")).unwrap();
        assert_eq!(low.upstream_model, "gemini-3.8-flash-low");
        let medium = resolve_route("gemini-3.8-flash", Some("medium")).unwrap();
        assert_eq!(medium.upstream_model, "gemini-3.8-flash-medium");
        let high = resolve_route("gemini-3.8-flash", Some("high")).unwrap();
        assert_eq!(high.upstream_model, "gemini-3.8-flash-high");
    }

    #[test]
    fn test_get_port_default() {
        assert!(get_port() > 0);
    }
}
