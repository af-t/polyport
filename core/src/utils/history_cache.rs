use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    pub thought_signature: String,
    pub reasoning: String,
}

static HISTORY_CACHE: LazyLock<Mutex<HashMap<String, HistoryEntry>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub const DUMMY_SIGNATURE: &str = "dummy_sig_from_proxy";
pub const FALLBACK_SIGNATURE: &str = "context_engineering_is_the_way_to_go";

fn is_usable_signature(sig: &str) -> bool {
    !sig.is_empty() && sig != DUMMY_SIGNATURE
}

pub fn record_tool_signature(tool_id: &str, signature: &str, reasoning: &str) {
    if tool_id.is_empty() || !is_usable_signature(signature) {
        return;
    }
    if let Ok(mut guard) = HISTORY_CACHE.lock() {
        guard.insert(
            tool_id.to_string(),
            HistoryEntry {
                thought_signature: signature.to_string(),
                reasoning: reasoning.to_string(),
            },
        );
    }
}

#[must_use]
pub fn lookup_tool_signature(tool_id: &str) -> Option<HistoryEntry> {
    if tool_id.is_empty() {
        return None;
    }
    HISTORY_CACHE
        .lock()
        .ok()
        .and_then(|guard| guard.get(tool_id).cloned())
        .filter(|entry| is_usable_signature(&entry.thought_signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_lookup_round_trip() {
        record_tool_signature("test-cache-id-1", "pxsig_live_abc", "some reasoning");
        let entry = lookup_tool_signature("test-cache-id-1").unwrap();
        assert_eq!(entry.thought_signature, "pxsig_live_abc");
        assert_eq!(entry.reasoning, "some reasoning");
    }

    #[test]
    fn test_dummy_and_empty_signatures_are_dropped() {
        record_tool_signature("test-cache-id-2", DUMMY_SIGNATURE, "");
        record_tool_signature("test-cache-id-3", "", "");
        assert_eq!(lookup_tool_signature("test-cache-id-2"), None);
        assert_eq!(lookup_tool_signature("test-cache-id-3"), None);
        assert_eq!(lookup_tool_signature("test-cache-id-unknown"), None);
    }
}
