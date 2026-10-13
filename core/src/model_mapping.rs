use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{ClaudeModelTier, ModelNotFoundError, RouteDecision, find_model_info, resolve_route};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MappingProvider {
    Google,
    Anthropic,
    Codex,
}

impl MappingProvider {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Anthropic => "anthropic",
            Self::Codex => "codex",
        }
    }

    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "google" => Some(Self::Google),
            "anthropic" => Some(Self::Anthropic),
            "codex" => Some(Self::Codex),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MappingEndpoint {
    Messages,
    #[serde(rename = "openai")]
    OpenAi,
}

impl MappingEndpoint {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::OpenAi => "openai",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMappingTarget {
    pub provider: MappingProvider,
    pub model: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadedModelMapping {
    pub include_openai: bool,
    pub targets: HashMap<ClaudeModelTier, Vec<ModelMappingTarget>>,
}

#[must_use]
pub fn mapping_path() -> PathBuf {
    if let Ok(config_home) = std::env::var("XDG_CONFIG_HOME")
        && !config_home.is_empty()
    {
        return PathBuf::from(config_home).join("polyport").join("model-mapping.json");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".config")
        .join("polyport")
        .join("model-mapping.json")
}

#[must_use]
pub fn read_mapping_from_path(path: &Path) -> LoadedModelMapping {
    let Ok(raw_text) = std::fs::read_to_string(path) else {
        return LoadedModelMapping::default();
    };

    let Ok(raw_val) = serde_json::from_str::<serde_json::Value>(&raw_text) else {
        tracing::warn!("[ModelMapping] invalid JSON; using defaults");
        return LoadedModelMapping::default();
    };

    let Some(obj) = raw_val.as_object() else {
        tracing::warn!("[ModelMapping] configuration root must be an object; using defaults");
        return LoadedModelMapping::default();
    };

    let include_openai = obj
        .get("include_openai")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let tiers = [
        ("opus", ClaudeModelTier::Opus),
        ("sonnet", ClaudeModelTier::Sonnet),
        ("haiku", ClaudeModelTier::Haiku),
    ];

    let mut targets = HashMap::new();

    for (tier_name, tier_enum) in tiers {
        let Some(configured) = obj.get(tier_name) else {
            continue;
        };

        let Some(arr) = configured.as_array() else {
            tracing::warn!("[ModelMapping] {tier_name} mapping must be an array; ignoring class");
            continue;
        };

        let mut valid_targets = Vec::new();

        for item in arr {
            let Some(item_obj) = item.as_object() else {
                tracing::warn!("[ModelMapping] invalid {tier_name} mapping target; ignoring target");
                continue;
            };

            let Some(provider_str) = item_obj.get("provider").and_then(serde_json::Value::as_str) else {
                tracing::warn!("[ModelMapping] invalid {tier_name} mapping target; ignoring target");
                continue;
            };

            let Some(provider) = MappingProvider::from_str_opt(provider_str) else {
                tracing::warn!("[ModelMapping] invalid {tier_name} mapping target; ignoring target");
                continue;
            };

            let Some(model_str) = item_obj.get("model").and_then(serde_json::Value::as_str) else {
                tracing::warn!("[ModelMapping] invalid {tier_name} mapping target; ignoring target");
                continue;
            };

            if model_str.is_empty() {
                tracing::warn!("[ModelMapping] invalid {tier_name} mapping target; ignoring target");
                continue;
            }

            let Some(model_info) = find_model_info(model_str) else {
                tracing::warn!("[ModelMapping] unsupported {tier_name} mapping model; ignoring target");
                continue;
            };

            if model_info.owned_by != provider.as_str() {
                tracing::warn!("[ModelMapping] provider does not match {tier_name} mapping model; ignoring target");
                continue;
            }

            valid_targets.push(ModelMappingTarget {
                provider,
                model: model_str.to_string(),
            });
        }

        if !valid_targets.is_empty() {
            targets.insert(tier_enum, valid_targets);
        }
    }

    LoadedModelMapping {
        include_openai,
        targets,
    }
}

#[must_use]
pub fn load_model_mapping() -> LoadedModelMapping {
    read_mapping_from_path(&mapping_path())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteCandidate {
    pub requested_model: String,
    pub route: RouteDecision,
    pub mapped: bool,
}

/// Resolves route candidates using an explicit mapping configuration.
///
/// # Errors
/// Returns `ModelNotFoundError` if `requested_model` or any mapped target cannot be resolved.
pub fn resolve_route_candidates_with_mapping(
    requested_model: &str,
    endpoint: MappingEndpoint,
    effort: Option<&str>,
    mapping: &LoadedModelMapping,
) -> Result<Vec<RouteCandidate>, ModelNotFoundError> {
    let original_route = resolve_route(requested_model, effort)?;
    let model_info = find_model_info(requested_model);
    let tier = model_info.and_then(|info| info.claude_tier);

    let Some(tier) = tier else {
        return Ok(vec![RouteCandidate {
            requested_model: requested_model.to_string(),
            route: original_route,
            mapped: false,
        }]);
    };

    if endpoint == MappingEndpoint::OpenAi && !mapping.include_openai {
        return Ok(vec![RouteCandidate {
            requested_model: requested_model.to_string(),
            route: original_route,
            mapped: false,
        }]);
    }

    let Some(configured_targets) = mapping.targets.get(&tier) else {
        return Ok(vec![RouteCandidate {
            requested_model: requested_model.to_string(),
            route: original_route,
            mapped: false,
        }]);
    };

    if configured_targets.is_empty() {
        return Ok(vec![RouteCandidate {
            requested_model: requested_model.to_string(),
            route: original_route,
            mapped: false,
        }]);
    }

    let mut candidates = Vec::with_capacity(configured_targets.len());
    for target in configured_targets {
        let route = resolve_route(&target.model, effort)?;
        candidates.push(RouteCandidate {
            requested_model: requested_model.to_string(),
            route,
            mapped: true,
        });
    }

    Ok(candidates)
}

/// Resolves route candidates using the default mapping file on disk.
///
/// # Errors
/// Returns `ModelNotFoundError` if `requested_model` or any mapped target cannot be resolved.
pub fn resolve_route_candidates(
    requested_model: &str,
    endpoint: MappingEndpoint,
    effort: Option<&str>,
) -> Result<Vec<RouteCandidate>, ModelNotFoundError> {
    let mapping = load_model_mapping();
    resolve_route_candidates_with_mapping(requested_model, endpoint, effort, &mapping)
}

#[must_use]
pub fn get_route_fingerprint(candidates: &[RouteCandidate], endpoint: MappingEndpoint) -> Option<String> {
    if !candidates.iter().any(|c| c.mapped) {
        return None;
    }

    let joined = candidates
        .iter()
        .map(|c| format!("{}:{}", c.route.provider.as_str(), c.route.upstream_model))
        .collect::<Vec<_>>()
        .join(",");

    Some(format!("{}|{}", endpoint.as_str(), joined))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("polyport-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_load_mapping_missing_file_returns_default() {
        let dir = TempDir::new();
        let non_existent = dir.path().join("model-mapping.json");
        let mapping = read_mapping_from_path(&non_existent);
        assert!(!mapping.include_openai);
        assert!(mapping.targets.is_empty());
    }

    #[test]
    fn test_load_mapping_valid_targets_and_include_openai() {
        let dir = TempDir::new();
        let path = dir.path().join("model-mapping.json");
        let json = serde_json::json!({
            "include_openai": true,
            "opus": [
                { "provider": "google", "model": "claude-opus-4-6-thinking" },
                { "provider": "codex", "model": "gpt-5.6-sol" }
            ]
        });
        fs::write(&path, json.to_string()).unwrap();

        let mapping = read_mapping_from_path(&path);
        assert!(mapping.include_openai);
        let opus_targets = mapping.targets.get(&ClaudeModelTier::Opus).unwrap();
        assert_eq!(opus_targets.len(), 2);
        assert_eq!(opus_targets[0].provider, MappingProvider::Google);
        assert_eq!(opus_targets[0].model, "claude-opus-4-6-thinking");
        assert_eq!(opus_targets[1].provider, MappingProvider::Codex);
        assert_eq!(opus_targets[1].model, "gpt-5.6-sol");
    }

    #[test]
    fn test_load_mapping_invalid_json_uses_default() {
        let dir = TempDir::new();
        let path = dir.path().join("model-mapping.json");
        fs::write(&path, "not-json").unwrap();
        let mapping = read_mapping_from_path(&path);
        assert!(!mapping.include_openai);
        assert!(mapping.targets.is_empty());

        fs::write(&path, "[]").unwrap();
        let mapping2 = read_mapping_from_path(&path);
        assert!(!mapping2.include_openai);
        assert!(mapping2.targets.is_empty());
    }

    #[test]
    fn test_load_mapping_drops_invalid_target_and_keeps_valid_sibling() {
        let dir = TempDir::new();
        let path = dir.path().join("model-mapping.json");
        let json = serde_json::json!({
            "sonnet": [
                { "provider": "not-a-provider", "model": "gpt-5.6-terra" },
                { "provider": "codex", "model": "gpt-5.6-terra" }
            ]
        });
        fs::write(&path, json.to_string()).unwrap();

        let mapping = read_mapping_from_path(&path);
        let sonnet_targets = mapping.targets.get(&ClaudeModelTier::Sonnet).unwrap();
        assert_eq!(sonnet_targets.len(), 1);
        assert_eq!(sonnet_targets[0].provider, MappingProvider::Codex);
        assert_eq!(sonnet_targets[0].model, "gpt-5.6-terra");
    }

    #[test]
    fn test_load_mapping_drops_unknown_model_and_mismatched_provider() {
        let dir = TempDir::new();
        let path = dir.path().join("model-mapping.json");
        let json = serde_json::json!({
            "opus": [
                { "provider": "google", "model": "not-a-model" },
                { "provider": "google", "model": "gpt-5.6-sol" },
                { "provider": "codex", "model": "gpt-5.6-sol" }
            ]
        });
        fs::write(&path, json.to_string()).unwrap();

        let mapping = read_mapping_from_path(&path);
        let opus_targets = mapping.targets.get(&ClaudeModelTier::Opus).unwrap();
        assert_eq!(opus_targets.len(), 1);
        assert_eq!(opus_targets[0].provider, MappingProvider::Codex);
        assert_eq!(opus_targets[0].model, "gpt-5.6-sol");
    }

    #[test]
    fn test_resolve_route_candidates_unmapped_for_non_claude_or_empty_mapping() {
        let mapping = LoadedModelMapping::default();
        let candidates =
            resolve_route_candidates_with_mapping("gpt-5.5", MappingEndpoint::Messages, None, &mapping).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].requested_model, "gpt-5.5");
        assert_eq!(candidates[0].route.provider, crate::types::Provider::Codex);
        assert!(!candidates[0].mapped);
    }

    #[test]
    fn test_resolve_route_candidates_mapped_claude_tier_preserves_requested_model() {
        let mut mapping = LoadedModelMapping::default();
        mapping.targets.insert(
            ClaudeModelTier::Opus,
            vec![
                ModelMappingTarget {
                    provider: MappingProvider::Google,
                    model: "claude-opus-4-6-thinking".to_string(),
                },
                ModelMappingTarget {
                    provider: MappingProvider::Codex,
                    model: "gpt-5.6-sol".to_string(),
                },
            ],
        );

        let candidates =
            resolve_route_candidates_with_mapping("claude-opus-4-8-native", MappingEndpoint::Messages, None, &mapping)
                .unwrap();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].requested_model, "claude-opus-4-8-native");
        assert_eq!(candidates[0].route.provider, crate::types::Provider::Gemini);
        assert_eq!(candidates[0].route.upstream_model, "claude-opus-4-6-thinking");
        assert!(candidates[0].mapped);

        assert_eq!(candidates[1].requested_model, "claude-opus-4-8-native");
        assert_eq!(candidates[1].route.provider, crate::types::Provider::Codex);
        assert_eq!(candidates[1].route.upstream_model, "gpt-5.6-sol");
        assert!(candidates[1].mapped);
    }

    #[test]
    fn test_resolve_route_candidates_openai_endpoint_honors_include_openai() {
        let mut mapping = LoadedModelMapping::default();
        mapping.targets.insert(
            ClaudeModelTier::Sonnet,
            vec![ModelMappingTarget {
                provider: MappingProvider::Codex,
                model: "gpt-5.6-terra".to_string(),
            }],
        );

        // include_openai is false by default
        let unmapped =
            resolve_route_candidates_with_mapping("claude-sonnet-5-native", MappingEndpoint::OpenAi, None, &mapping)
                .unwrap();
        assert_eq!(unmapped.len(), 1);
        assert!(!unmapped[0].mapped);
        assert_eq!(unmapped[0].route.provider, crate::types::Provider::Claude);

        // explicitly true
        mapping.include_openai = true;
        let mapped =
            resolve_route_candidates_with_mapping("claude-sonnet-5-native", MappingEndpoint::OpenAi, None, &mapping)
                .unwrap();
        assert_eq!(mapped.len(), 1);
        assert!(mapped[0].mapped);
        assert_eq!(mapped[0].route.provider, crate::types::Provider::Codex);
    }

    #[test]
    fn test_get_route_fingerprint_mapped_and_unmapped() {
        let mut mapping = LoadedModelMapping::default();
        mapping.targets.insert(
            ClaudeModelTier::Opus,
            vec![
                ModelMappingTarget {
                    provider: MappingProvider::Google,
                    model: "claude-opus-4-6-thinking".to_string(),
                },
                ModelMappingTarget {
                    provider: MappingProvider::Codex,
                    model: "gpt-5.6-sol".to_string(),
                },
            ],
        );

        let mapped =
            resolve_route_candidates_with_mapping("claude-opus-4-8-native", MappingEndpoint::Messages, None, &mapping)
                .unwrap();
        let fp = get_route_fingerprint(&mapped, MappingEndpoint::Messages);
        assert_eq!(
            fp.as_deref(),
            Some("messages|gemini:claude-opus-4-6-thinking,codex:gpt-5.6-sol")
        );

        let unmapped =
            resolve_route_candidates_with_mapping("gpt-5.5", MappingEndpoint::Messages, None, &mapping).unwrap();
        assert_eq!(get_route_fingerprint(&unmapped, MappingEndpoint::Messages), None);
    }
}
