use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelModalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub provider: String,
    #[serde(default)]
    pub featured: bool,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub effort: Vec<String>,
    #[serde(default)]
    pub modalities: Option<ModelModalities>,
}

pub const MODELS_JSON: &str = include_str!("models.json");

#[must_use]
pub fn models_catalog_json() -> &'static str {
    MODELS_JSON
}

static CATALOG: OnceLock<Vec<ModelDescriptor>> = OnceLock::new();

/// Returns parsed model descriptors from embedded catalog.
///
/// # Errors
/// Returns error if embedded JSON is malformed.
pub fn get_models_catalog() -> Result<&'static [ModelDescriptor], serde_json::Error> {
    if let Some(catalog) = CATALOG.get() {
        return Ok(catalog.as_slice());
    }
    let parsed: Vec<ModelDescriptor> = serde_json::from_str(MODELS_JSON)?;
    let _ = CATALOG.set(parsed);
    match CATALOG.get() {
        Some(catalog) => Ok(catalog.as_slice()),
        None => Ok(&[]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_catalog_parses_embedded_json() {
        let catalog = get_models_catalog().unwrap();
        assert_ne!(catalog.len(), 0);
        assert!(models_catalog_json().starts_with('['));
    }

    #[test]
    fn test_models_catalog_contains_expected_models() {
        let catalog = get_models_catalog().unwrap();
        let ids: Vec<&str> = catalog.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"gemini-3.5-flash"));
        assert!(ids.contains(&"claude-sonnet-5-native"));
        assert!(ids.contains(&"gpt-5.6-sol"));
    }
}
