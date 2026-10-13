use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{ClaudeModelTier, find_model_info};

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
}
