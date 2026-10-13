use serde_json::Value;
use sha2::{Digest, Sha256};

fn uuid_from_hash(hash: &str) -> String {
    if hash.len() < 32 {
        return String::new();
    }
    format!(
        "{}-{}-4{}-8{}-{}",
        &hash[0..8],
        &hash[8..12],
        &hash[13..16],
        &hash[17..20],
        &hash[20..32]
    )
}

use std::fmt::Write;

fn to_hex_digest(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Derives a stable per-host device ID formatted as an RFC-4122 UUID.
#[must_use]
pub fn get_device_id() -> String {
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
        .unwrap_or_else(|_| "localhost".to_string());
    let mut hasher = Sha256::new();
    hasher.update(hostname.as_bytes());
    let hex_digest = to_hex_digest(&hasher.finalize());
    uuid_from_hash(&hex_digest)
}

/// Derives a deterministic session ID from model, system message, and first message.
#[must_use]
pub fn get_deterministic_session_id(body: &Value) -> String {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");

    let messages = body.get("messages").and_then(Value::as_array);

    let first_msg = messages.and_then(|m| m.first());
    let first_content = first_msg
        .and_then(|m| m.get("content"))
        .map_or_else(String::new, |c| match c {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });

    let system_msg = messages.and_then(|m| {
        m.iter().find(|msg| {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
            role == "system" || role == "developer"
        })
    });
    let system_content = system_msg
        .and_then(|m| m.get("content"))
        .map_or_else(String::new, |c| match c {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });

    let input = format!("{model}|{system_content}|{first_content}");
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hex_digest = to_hex_digest(&hasher.finalize());
    uuid_from_hash(&hex_digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_uuid_from_hash_formatting() {
        let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let uuid = uuid_from_hash(hash);
        assert_eq!(uuid.len(), 36);
        let parts: Vec<&str> = uuid.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], "01234567");
        assert_eq!(parts[1], "89ab");
        assert_eq!(parts[2], "4def");
        assert_eq!(parts[3], "8123");
        assert_eq!(parts[4], "456789abcdef");
    }

    #[test]
    fn test_get_device_id_stable() {
        let id1 = get_device_id();
        let id2 = get_device_id();
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 36);
        assert_eq!(id1.chars().nth(14), Some('4'));
        assert_eq!(id1.chars().nth(19), Some('8'));
    }

    #[test]
    fn test_get_deterministic_session_id_matches() {
        let body1 = json!({
            "model": "claude-sonnet-5",
            "messages": [
                { "role": "system", "content": "You are helpful" },
                { "role": "user", "content": "Hello" }
            ]
        });
        let body2 = json!({
            "model": "claude-sonnet-5",
            "messages": [
                { "role": "system", "content": "You are helpful" },
                { "role": "user", "content": "Hello" },
                { "role": "assistant", "content": "Hi there" }
            ]
        });
        let body3 = json!({
            "model": "claude-opus-5",
            "messages": [
                { "role": "system", "content": "You are helpful" },
                { "role": "user", "content": "Hello" }
            ]
        });

        let id1 = get_deterministic_session_id(&body1);
        let id2 = get_deterministic_session_id(&body2);
        let id3 = get_deterministic_session_id(&body3);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(id1.len(), 36);
    }
}
