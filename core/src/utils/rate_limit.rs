use reqwest::header::HeaderMap;
use thiserror::Error;

pub const DEFAULT_COOLDOWN_MS: u64 = 60_000;
pub const MAX_COOLDOWN_MS: u64 = 24 * 60 * 60 * 1000;

#[derive(Debug, Error)]
#[error("HTTP {status} rate limit, cooldown {cooldown_ms}ms")]
pub struct RateLimitError {
    pub status: u16,
    pub cooldown_ms: u64,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct NonAccountError {
    pub status: u16,
    message: String,
}

impl NonAccountError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct UpstreamError {
    pub status: u16,
    message: String,
}

impl UpstreamError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

#[must_use]
pub fn is_rate_limit_status(status: u16, message: &str) -> bool {
    if status == 429 {
        return true;
    }
    let lower = message.to_lowercase();
    lower.contains("quota")
        || lower.contains("rate limit")
        || lower.contains("ratelimit")
        || lower.contains("exhausted")
}

const FLOOR_MS: f64 = 1_000.0;
const CEIL_MS: f64 = 86_400_000.0;

fn f64_to_u64_in_range(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    value.round().to_string().parse::<u64>().ok()
}

fn clamp_finite(raw_ms: f64) -> Option<u64> {
    if !raw_ms.is_finite() {
        return None;
    }
    if raw_ms <= FLOOR_MS {
        return Some(1_000);
    }
    if raw_ms >= CEIL_MS {
        return Some(MAX_COOLDOWN_MS);
    }
    Some(f64_to_u64_in_range(raw_ms).unwrap_or(MAX_COOLDOWN_MS))
}

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

fn parse_unsigned_decimal(text: &str) -> Option<f64> {
    let (int_part, frac_part) = match text.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (text, None),
    };
    if int_part.is_empty() || !int_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if let Some(frac) = frac_part
        && (frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    text.parse::<f64>().ok()
}

fn strip_unit<'a>(text: &'a str, unit: &str) -> Option<&'a str> {
    if text.len() >= unit.len() && text[text.len() - unit.len()..].eq_ignore_ascii_case(unit) {
        Some(text[..text.len() - unit.len()].trim_end())
    } else {
        None
    }
}

fn parse_delay_str(text: &str) -> Option<f64> {
    let text = text.trim();
    let (number, unit_ms) = if let Some(rest) = strip_unit(text, "ms") {
        (rest, 1.0)
    } else if let Some(rest) = strip_unit(text, "s") {
        (rest, 1_000.0)
    } else {
        (text, 1_000.0)
    };
    parse_unsigned_decimal(number).map(|v| v * unit_ms)
}

fn number_as_f64(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn retry_delay_value_ms(raw: &serde_json::Value) -> Option<f64> {
    match raw {
        serde_json::Value::Number(n) => {
            let v = n.as_f64()?;
            v.is_finite().then_some(v * 1_000.0)
        }
        serde_json::Value::String(s) => parse_delay_str(s),
        serde_json::Value::Object(map) => {
            if !map.contains_key("seconds") && !map.contains_key("nanos") {
                return None;
            }
            let seconds = map.get("seconds").and_then(number_as_f64).unwrap_or(0.0);
            let nanos = map.get("nanos").and_then(number_as_f64).unwrap_or(0.0);
            if !seconds.is_finite() || !nanos.is_finite() {
                return None;
            }
            Some(seconds * 1_000.0 + nanos / 1_000_000.0)
        }
        _ => None,
    }
}

fn gemini_retry_delay_ms(body: &str) -> Option<u64> {
    let parsed: serde_json::Value = serde_json::from_str(body).ok()?;
    let details = parsed.get("error")?.get("details")?;
    let list: Vec<&serde_json::Value> = match details {
        serde_json::Value::Array(items) => items.iter().collect(),
        obj if obj.is_object() => vec![obj],
        _ => return None,
    };
    for detail in list {
        let Some(raw) = detail.get("retryDelay").or_else(|| detail.get("retry_delay")) else {
            continue;
        };
        if let Some(ms) = retry_delay_value_ms(raw).and_then(clamp_finite) {
            return Some(ms);
        }
    }
    None
}

#[must_use]
pub fn parse_cooldown_ms(headers: &HeaderMap, body: &str, now_ms: u64) -> u64 {
    if let Some(retry_after) = header_str(headers, "retry-after") {
        let trimmed = retry_after.trim();
        if let Ok(secs) = trimmed.parse::<f64>()
            && let Some(ms) = clamp_finite(secs * 1_000.0)
        {
            return ms;
        }
        if let Ok(when) = httpdate::parse_http_date(trimmed)
            && let Ok(elapsed) = when.duration_since(std::time::UNIX_EPOCH)
            && let Ok(date_ms) = u64::try_from(elapsed.as_millis())
        {
            return date_ms.saturating_sub(now_ms).clamp(1_000, MAX_COOLDOWN_MS);
        }
    }

    if let Some(reset) = header_str(headers, "anthropic-ratelimit-unified-reset")
        && let Ok(epoch_secs) = reset.trim().parse::<f64>()
        && epoch_secs.is_finite()
    {
        if epoch_secs <= 0.0 {
            return 1_000;
        }
        match f64_to_u64_in_range((epoch_secs * 1_000.0).round()) {
            Some(reset_ms) => {
                return reset_ms.saturating_sub(now_ms).clamp(1_000, MAX_COOLDOWN_MS);
            }
            None => return MAX_COOLDOWN_MS,
        }
    }

    if let Some(ms) = gemini_retry_delay_ms(body) {
        return ms;
    }

    DEFAULT_COOLDOWN_MS
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderName};

    const NOW: u64 = 1_000_000_000_000;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(k.parse::<HeaderName>().unwrap(), v.parse().unwrap());
        }
        h
    }

    #[test]
    fn rate_limit_error_carries_status_and_cooldown() {
        let e = RateLimitError {
            status: 429,
            cooldown_ms: 5000,
        };
        assert_eq!(e.status, 429);
        assert_eq!(e.cooldown_ms, 5000);
    }

    #[test]
    fn status_429_is_always_rate_limit() {
        assert!(is_rate_limit_status(429, ""));
    }

    #[test]
    fn message_keywords_count_on_other_statuses() {
        assert!(is_rate_limit_status(400, "RESOURCE_EXHAUSTED: quota"));
        assert!(is_rate_limit_status(500, "rate limit reached"));
    }

    #[test]
    fn plain_errors_are_not_rate_limits() {
        assert!(!is_rate_limit_status(400, "bad request"));
        assert!(!is_rate_limit_status(500, "boom"));
    }

    #[test]
    fn non_account_error_defaults_to_503() {
        let e = NonAccountError::new("socket cap reached");
        assert_eq!(e.status, 503);
    }

    #[test]
    fn upstream_error_carries_status() {
        let e = UpstreamError::new(400, "bad request");
        assert_eq!(e.status, 400);
    }

    #[test]
    fn retry_after_seconds() {
        let h = headers(&[("retry-after", "30")]);
        assert_eq!(parse_cooldown_ms(&h, "", NOW), 30_000);
    }

    #[test]
    fn retry_after_http_date() {
        let date = httpdate::fmt_http_date(std::time::UNIX_EPOCH + std::time::Duration::from_millis(NOW + 45_000));
        let h = headers(&[("retry-after", date.as_str())]);
        let ms = parse_cooldown_ms(&h, "", NOW);
        assert!((44_000..=45_000).contains(&ms));
    }

    #[test]
    fn anthropic_unified_reset_epoch_seconds() {
        let reset = ((NOW + 120_000) / 1000).to_string();
        let h = headers(&[("anthropic-ratelimit-unified-reset", reset.as_str())]);
        assert_eq!(parse_cooldown_ms(&h, "", NOW), 120_000);
    }

    #[test]
    fn gemini_retry_info_string_seconds() {
        let body = r#"{"error":{"details":[{"retryDelay":"17s"}]}}"#;
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), body, NOW), 17_000);
    }

    #[test]
    fn gemini_retry_info_duration_object() {
        let body = r#"{"error":{"details":[{"retryDelay":{"seconds":30,"nanos":500000000}}]}}"#;
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), body, NOW), 30_500);
    }

    #[test]
    fn gemini_retry_info_bare_numeric_string_is_seconds() {
        let body = r#"{"error":{"details":[{"retryDelay":"30"}]}}"#;
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), body, NOW), 30_000);
    }

    #[test]
    fn gemini_retry_info_snake_case_milliseconds() {
        let body = r#"{"error":{"details":[{"retry_delay":"1500ms"}]}}"#;
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), body, NOW), 1500);
    }

    #[test]
    fn gemini_retry_info_without_type_and_numeric_delay() {
        let body = r#"{"error":{"details":[{"retryDelay":5}]}}"#;
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), body, NOW), 5000);
    }

    #[test]
    fn default_when_nothing_present() {
        assert_eq!(parse_cooldown_ms(&HeaderMap::new(), "", NOW), DEFAULT_COOLDOWN_MS);
    }

    #[test]
    fn clamps_to_1s_floor_and_24h_ceiling() {
        let past = ((NOW - 99_000) / 1000).to_string();
        let past = headers(&[("anthropic-ratelimit-unified-reset", past.as_str())]);
        assert_eq!(parse_cooldown_ms(&past, "", NOW), 1000);
        let huge = headers(&[("retry-after", "864000")]);
        assert_eq!(parse_cooldown_ms(&huge, "", NOW), MAX_COOLDOWN_MS);
    }
}
