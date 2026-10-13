use crate::services::credentials::{ClaudeAccount, Credentials, claude_accounts_in};
use crate::types::{ModelClass, Provider};
use crate::utils::rate_limit::{NonAccountError, RateLimitError, UpstreamError};
use crate::utils::session_key::{last_assistant_from_messages, lookup_keys};
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

pub const AFFINITY_TTL_MS: u64 = 12 * 60 * 60 * 1000;
pub const AFFINITY_CAP: usize = 10_000;
pub const LONG_COOLDOWN_THRESHOLD_MS: u64 = 12 * 60 * 60 * 1000;

#[derive(Debug, Clone)]
pub struct AccountRuntime {
    pub id: String,
    pub cooldown_by_class: HashMap<ModelClass, u64>,
    pub last_used_seq: u64,
}

struct PinEntry {
    value: String,
    last_seen: u64,
    seq: u64,
}

struct PinStore {
    map: HashMap<String, PinEntry>,
    seq: u64,
}

impl PinStore {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            seq: 0,
        }
    }

    fn record(&mut self, key: &str, value: &str, now: u64) {
        self.seq += 1;
        let seq = self.seq;
        self.map.insert(
            key.to_owned(),
            PinEntry {
                value: value.to_owned(),
                last_seen: now,
                seq,
            },
        );
        while self.map.len() > AFFINITY_CAP {
            let Some(oldest) = self.map.iter().min_by_key(|(_, e)| e.seq).map(|(k, _)| k.clone()) else {
                break;
            };
            self.map.remove(&oldest);
        }
    }

    fn get(&mut self, key: &str, now: u64) -> Option<String> {
        let entry = self.map.get(key)?;
        if now.saturating_sub(entry.last_seen) > AFFINITY_TTL_MS {
            self.map.remove(key);
            return None;
        }
        let value = self.map.get(key)?.value.clone();
        self.seq += 1;
        let seq = self.seq;
        if let Some(entry) = self.map.get_mut(key) {
            entry.last_seen = now;
            entry.seq = seq;
        }
        Some(value)
    }

    fn clear_for_value(&mut self, value: &str) {
        self.map.retain(|_, entry| entry.value != value);
    }
}

fn system_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

struct ProviderPool {
    runtimes: HashMap<String, AccountRuntime>,
    order: Vec<String>,
}

impl ProviderPool {
    fn new() -> Self {
        Self {
            runtimes: HashMap::new(),
            order: Vec::new(),
        }
    }
}

pub struct AccountPool {
    pools: HashMap<Provider, ProviderPool>,
    affinity: HashMap<Provider, PinStore>,
    conversation_identity: HashMap<Provider, PinStore>,
    clock: fn() -> u64,
    use_counter: u64,
}

impl Default for AccountPool {
    fn default() -> Self {
        Self::new()
    }
}

impl AccountPool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: HashMap::from([
                (Provider::Gemini, ProviderPool::new()),
                (Provider::Claude, ProviderPool::new()),
                (Provider::Codex, ProviderPool::new()),
            ]),
            affinity: HashMap::from([
                (Provider::Gemini, PinStore::new()),
                (Provider::Claude, PinStore::new()),
                (Provider::Codex, PinStore::new()),
            ]),
            conversation_identity: HashMap::from([
                (Provider::Gemini, PinStore::new()),
                (Provider::Claude, PinStore::new()),
                (Provider::Codex, PinStore::new()),
            ]),
            clock: system_epoch_ms,
            use_counter: 0,
        }
    }

    pub fn set_clock(&mut self, clock: fn() -> u64) {
        self.clock = clock;
    }

    fn now(&self) -> u64 {
        (self.clock)()
    }

    pub fn sync_accounts(&mut self, provider: Provider, ids: &[impl AsRef<str>]) {
        let Some(pool) = self.pools.get_mut(&provider) else {
            return;
        };
        for id in ids {
            let id = id.as_ref();
            pool.runtimes.entry(id.to_owned()).or_insert_with(|| AccountRuntime {
                id: id.to_owned(),
                cooldown_by_class: HashMap::new(),
                last_used_seq: 0,
            });
        }
        pool.runtimes
            .retain(|id, _| ids.iter().any(|wanted| wanted.as_ref() == id));
        pool.order = ids.iter().map(|id| id.as_ref().to_owned()).collect();
    }

    #[must_use]
    pub fn available_ids(&self, provider: Provider, class: ModelClass) -> Vec<String> {
        let now = self.now();
        let Some(pool) = self.pools.get(&provider) else {
            return Vec::new();
        };
        let index: HashMap<&str, usize> = pool.order.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
        let mut runtimes: Vec<&AccountRuntime> = pool
            .runtimes
            .values()
            .filter(|rt| {
                rt.cooldown_by_class.get(&class).is_none_or(|until| *until <= now) && index.contains_key(rt.id.as_str())
            })
            .collect();
        runtimes.sort_by_key(|rt| (rt.last_used_seq, index.get(rt.id.as_str()).copied()));
        runtimes.into_iter().map(|rt| rt.id.clone()).collect()
    }

    pub fn mark_cooldown(&mut self, provider: Provider, id: &str, class: ModelClass, until_ms: u64) {
        if let Some(rt) = self.pools.get_mut(&provider).and_then(|p| p.runtimes.get_mut(id)) {
            rt.cooldown_by_class.insert(class, until_ms);
        }
    }

    pub fn record_affinity(&mut self, provider: Provider, key: &str, account_id: &str) {
        let now = self.now();
        if let Some(store) = self.affinity.get_mut(&provider) {
            if let Some(prev) = store.map.get(key).map(|entry| entry.value.clone())
                && prev != account_id
            {
                tracing::debug!("[Affinity] re-pin {provider:?} {key}: {prev} -> {account_id}");
            }
            store.record(key, account_id, now);
        }
    }

    pub fn get_affinity(&mut self, provider: Provider, key: &str) -> Option<String> {
        let now = self.now();
        self.affinity.get_mut(&provider)?.get(key, now)
    }

    pub fn resolve_account_pin(&mut self, provider: Provider, keys: &[impl AsRef<str>]) -> Option<String> {
        let now = self.now();
        let store = self.affinity.get_mut(&provider)?;
        for key in keys {
            if let Some(hit) = store.get(key.as_ref(), now) {
                return Some(hit);
            }
        }
        None
    }

    pub fn record_account_pin(&mut self, provider: Provider, keys: &[impl AsRef<str>], account_id: Option<&str>) {
        let Some(account_id) = account_id else {
            return;
        };
        for key in keys {
            self.record_affinity(provider, key.as_ref(), account_id);
        }
    }

    pub fn clear_affinity_for_account(&mut self, provider: Provider, account_id: &str) {
        if let Some(store) = self.affinity.get_mut(&provider) {
            store.clear_for_value(account_id);
        }
    }

    pub fn record_conversation_id(&mut self, provider: Provider, keys: &[impl AsRef<str>], id: Option<&str>) {
        let Some(id) = id else {
            return;
        };
        let now = self.now();
        if let Some(store) = self.conversation_identity.get_mut(&provider) {
            for key in keys {
                store.record(key.as_ref(), id, now);
            }
        }
    }

    pub fn resolve_conversation_id(&mut self, provider: Provider, keys: &[impl AsRef<str>]) -> Option<String> {
        let now = self.now();
        let store = self.conversation_identity.get_mut(&provider)?;
        for key in keys {
            if let Some(hit) = store.get(key.as_ref(), now) {
                return Some(hit);
            }
        }
        None
    }
}

#[must_use]
pub fn is_long_cooldown(ms: u64) -> bool {
    ms >= LONG_COOLDOWN_THRESHOLD_MS
}

#[derive(Debug, Error)]
pub enum AttemptError {
    #[error(transparent)]
    RateLimit(#[from] RateLimitError),
    #[error(transparent)]
    NonAccount(#[from] NonAccountError),
    #[error(transparent)]
    Upstream(#[from] UpstreamError),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AttemptError {
    fn status(&self) -> Option<u16> {
        match self {
            Self::RateLimit(e) => Some(e.status),
            Self::NonAccount(e) => Some(e.status),
            Self::Upstream(e) => Some(e.status),
            Self::Other(_) => None,
        }
    }

    fn is_deterministic_client_error(&self) -> bool {
        match self.status() {
            Some(status) => (400..500).contains(&status) && ![401, 403, 408, 429].contains(&status),
            None => false,
        }
    }

    fn is_non_account_failure(&self) -> bool {
        matches!(self, Self::NonAccount(_))
    }

    fn cooldown_ms(&self) -> u64 {
        match self {
            Self::RateLimit(e) => e.cooldown_ms,
            _ => 10_000,
        }
    }
}

fn rate_limited(cooldown_ms: u64) -> AttemptError {
    AttemptError::RateLimit(RateLimitError {
        status: 429,
        cooldown_ms,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinSource {
    Header,
    Token,
    Tail,
}

#[derive(Debug, Clone)]
pub struct PreferredAccount {
    pub account_id: Option<String>,
    pub lookup_keys: Vec<String>,
    pub pin_source: Option<PinSource>,
}

#[derive(Debug, Clone)]
pub struct ClaudeSession {
    pub accounts: Vec<ClaudeAccount>,
    pub account_id: Option<String>,
    pub lookup_keys: Vec<String>,
    pub pin_source: Option<PinSource>,
}

impl AccountPool {
    fn stamp_use(&mut self, provider: Provider, id: &str) {
        self.use_counter += 1;
        let seq = self.use_counter;
        if let Some(rt) = self.pools.get_mut(&provider).and_then(|pool| pool.runtimes.get_mut(id)) {
            rt.last_used_seq = seq;
        }
    }

    async fn failover_unpinned<T>(
        &mut self,
        provider: Provider,
        class: ModelClass,
        mut attempt: impl AsyncFnMut(&str) -> Result<T, AttemptError>,
    ) -> Result<T, AttemptError> {
        let candidates = self.available_ids(provider, class);
        if candidates.is_empty() {
            return Err(rate_limited(0));
        }
        let mut last_err: Option<AttemptError> = None;
        for id in candidates {
            self.stamp_use(provider, &id);
            match attempt(id.as_str()).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if err.is_deterministic_client_error() || err.is_non_account_failure() {
                        return Err(err);
                    }
                    let cooldown_ms = err.cooldown_ms();
                    let until = self.now() + cooldown_ms;
                    self.mark_cooldown(provider, &id, class, until);
                    tracing::warn!(
                        "[Proxy] Account {id} failed with error: {err}. Cooling down for {cooldown_ms}ms and trying next candidate."
                    );
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| rate_limited(0)))
    }

    /// # Errors
    /// Returns the attempt error directly, or a 429 [`AttemptError::RateLimit`] when no account
    /// is available.
    pub async fn with_failover<T>(
        &mut self,
        provider: Provider,
        class: ModelClass,
        preferred: Option<&str>,
        mut attempt: impl AsyncFnMut(&str) -> Result<T, AttemptError>,
    ) -> Result<T, AttemptError> {
        if let Some(preferred) = preferred {
            let now = self.now();
            let known = self
                .pools
                .get(&provider)
                .is_some_and(|pool| pool.runtimes.contains_key(preferred));
            if !known {
                return Err(rate_limited(0));
            }
            let cooldown_until = self
                .pools
                .get(&provider)
                .and_then(|pool| pool.runtimes.get(preferred))
                .and_then(|rt| rt.cooldown_by_class.get(&class).copied())
                .unwrap_or(0);
            if cooldown_until > now {
                if is_long_cooldown(cooldown_until - now) {
                    tracing::warn!(
                        "[Proxy] Pinned {provider:?} account {preferred} has a long cooldown. Unpinning and falling over to next available candidate."
                    );
                    self.clear_affinity_for_account(provider, preferred);
                } else {
                    return Err(rate_limited(cooldown_until - now));
                }
            } else {
                self.stamp_use(provider, preferred);
                match attempt(preferred).await {
                    Ok(value) => return Ok(value),
                    Err(err) => {
                        if err.is_deterministic_client_error() || err.is_non_account_failure() {
                            tracing::warn!(
                                "[Proxy] Pinned account {preferred} failed with error: {err}. Session is pinned - no failover, propagating to client."
                            );
                            return Err(err);
                        }
                        let cooldown_ms = err.cooldown_ms();
                        let until = self.now() + cooldown_ms;
                        self.mark_cooldown(provider, preferred, class, until);
                        if is_long_cooldown(cooldown_ms) {
                            tracing::warn!(
                                "[Proxy] Pinned account {preferred} hit a long rate limit. Unpinning and falling over to next available candidate."
                            );
                            self.clear_affinity_for_account(provider, preferred);
                        } else {
                            tracing::warn!(
                                "[Proxy] Pinned account {preferred} failed with error: {err}. Session is pinned - no failover, propagating to client."
                            );
                            return Err(err);
                        }
                    }
                }
            }
        }
        self.failover_unpinned(provider, class, attempt).await
    }

    pub fn resolve_preferred_account(
        &mut self,
        provider: Provider,
        headers: &HeaderMap,
        messages: &Value,
    ) -> PreferredAccount {
        let anchor = last_assistant_from_messages(messages);
        let has_anchor = anchor.token.is_some() || anchor.text.is_some();
        let header = if has_anchor {
            headers.get("x-claude-code-session-id").and_then(|v| v.to_str().ok())
        } else {
            None
        };
        let keys = lookup_keys(header, anchor.token.as_deref(), anchor.text.as_deref());
        let mut account_id = None;
        let mut pin_source = None;
        for key in &keys {
            if let Some(hit) = self.get_affinity(provider, key) {
                account_id = Some(hit);
                pin_source = Some(if key.starts_with("hdr:") {
                    PinSource::Header
                } else if key.starts_with("tok:") {
                    PinSource::Token
                } else {
                    PinSource::Tail
                });
                break;
            }
        }
        tracing::debug!(
            "[SessionPin] provider={provider:?} source={pin_source:?} anchor={} preferred={}",
            if has_anchor { "yes" } else { "no" },
            account_id.as_deref().unwrap_or("none"),
        );
        PreferredAccount {
            account_id,
            lookup_keys: keys,
            pin_source,
        }
    }

    pub fn resolve_claude_session(
        &mut self,
        headers: &HeaderMap,
        messages: &Value,
        creds: &Credentials,
    ) -> ClaudeSession {
        let accounts: Vec<ClaudeAccount> = claude_accounts_in(creds)
            .into_iter()
            .filter(|account| !account.claude_ai_oauth.access_token.is_empty())
            .collect();
        let ids: Vec<&str> = accounts.iter().map(|account| account.id.as_str()).collect();
        self.sync_accounts(Provider::Claude, &ids);
        let preferred = self.resolve_preferred_account(Provider::Claude, headers, messages);
        ClaudeSession {
            accounts,
            account_id: preferred.account_id,
            lookup_keys: preferred.lookup_keys,
            pin_source: preferred.pin_source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelClass, Provider};

    #[test]
    fn synced_accounts_start_available_in_config_order() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b", "c"]);
        assert_eq!(
            pool.available_ids(Provider::Gemini, ModelClass::GeminiNative),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn removing_an_id_drops_it() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        pool.sync_accounts(Provider::Gemini, &["a"]);
        assert_eq!(
            pool.available_ids(Provider::Gemini, ModelClass::GeminiNative),
            vec!["a"]
        );
    }

    #[test]
    fn cooled_account_excluded_until_time_passes() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Claude, &["a", "b"]);
        pool.mark_cooldown(Provider::Claude, "a", ModelClass::Claude, 6000);
        assert_eq!(pool.available_ids(Provider::Claude, ModelClass::Claude), vec!["b"]);
        pool.set_clock(|| 6000);
        let mut ids = pool.available_ids(Provider::Claude, ModelClass::Claude);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn cooldown_isolated_per_model_class() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        pool.mark_cooldown(Provider::Gemini, "a", ModelClass::Claude, 6000);
        assert_eq!(pool.available_ids(Provider::Gemini, ModelClass::Claude), vec!["b"]);
        let mut ids = pool.available_ids(Provider::Gemini, ModelClass::GeminiNative);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn affinity_record_get_is_per_provider() {
        let mut pool = AccountPool::new();
        pool.record_affinity(Provider::Claude, "tail:x", "acc1");
        assert_eq!(pool.get_affinity(Provider::Claude, "tail:x").as_deref(), Some("acc1"));
        assert_eq!(pool.get_affinity(Provider::Gemini, "tail:x"), None);
    }

    #[test]
    fn affinity_entry_expires_after_ttl_but_not_at_boundary() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.record_affinity(Provider::Claude, "k-boundary", "acc1");
        pool.record_affinity(Provider::Claude, "k-expired", "acc1");
        pool.set_clock(|| 1000 + AFFINITY_TTL_MS);
        assert_eq!(
            pool.get_affinity(Provider::Claude, "k-boundary").as_deref(),
            Some("acc1")
        );
        pool.set_clock(|| 1000 + AFFINITY_TTL_MS + 1);
        assert_eq!(pool.get_affinity(Provider::Claude, "k-expired"), None);
    }

    #[test]
    fn affinity_capacity_evicts_oldest() {
        let mut pool = AccountPool::new();
        for i in 0..10_001 {
            pool.record_affinity(Provider::Gemini, &format!("k{i}"), "acc");
        }
        assert_eq!(pool.get_affinity(Provider::Gemini, "k0"), None);
        assert_eq!(pool.get_affinity(Provider::Gemini, "k10000").as_deref(), Some("acc"));
    }

    #[test]
    fn conversation_identity_resolves_via_any_key() {
        let mut pool = AccountPool::new();
        pool.record_conversation_id(Provider::Codex, &["hdr:s1", "tok:a", "tail:b"], Some("conv1"));
        assert_eq!(
            pool.resolve_conversation_id(Provider::Codex, &["tok:a"]).as_deref(),
            Some("conv1")
        );
        assert_eq!(
            pool.resolve_conversation_id(Provider::Codex, &["tail:b"]).as_deref(),
            Some("conv1")
        );
        assert_eq!(pool.resolve_conversation_id(Provider::Codex, &["tok:c"]), None);
        assert_eq!(pool.resolve_conversation_id(Provider::Gemini, &["tok:a"]), None);
    }

    #[test]
    fn conversation_identity_expires_after_ttl() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.record_conversation_id(Provider::Codex, &["k"], Some("conv1"));
        pool.set_clock(|| 1000 + AFFINITY_TTL_MS + 1);
        assert_eq!(pool.resolve_conversation_id(Provider::Codex, &["k"]), None);
    }

    #[test]
    fn conversation_identity_capacity_evicts_oldest() {
        let mut pool = AccountPool::new();
        for i in 0..10_001 {
            pool.record_conversation_id(Provider::Codex, &[format!("k{i}")], Some("conv"));
        }
        assert_eq!(pool.resolve_conversation_id(Provider::Codex, &["k0"]), None);
        assert_eq!(
            pool.resolve_conversation_id(Provider::Codex, &["k10000"]).as_deref(),
            Some("conv")
        );
    }

    #[test]
    fn resolve_account_pin_checks_keys_in_order() {
        let mut pool = AccountPool::new();
        pool.record_affinity(Provider::Gemini, "tok:a", "acc-a");
        pool.record_affinity(Provider::Gemini, "tail:b", "acc-b");
        assert_eq!(
            pool.resolve_account_pin(Provider::Gemini, &["tok:a", "tail:b"])
                .as_deref(),
            Some("acc-a")
        );
        assert_eq!(
            pool.resolve_account_pin(Provider::Gemini, &["tok:miss", "tail:b"])
                .as_deref(),
            Some("acc-b")
        );
        assert_eq!(pool.resolve_account_pin(Provider::Gemini, &["tok:miss"]), None);
    }

    #[test]
    fn record_account_pin_no_ops_on_none() {
        let mut pool = AccountPool::new();
        pool.record_account_pin(Provider::Gemini, &["k"], None);
        assert_eq!(pool.get_affinity(Provider::Gemini, "k"), None);
    }

    #[test]
    fn clear_affinity_removes_only_that_account() {
        let mut pool = AccountPool::new();
        pool.record_affinity(Provider::Gemini, "k1", "a");
        pool.record_affinity(Provider::Gemini, "k2", "b");
        pool.clear_affinity_for_account(Provider::Gemini, "a");
        assert_eq!(pool.get_affinity(Provider::Gemini, "k1"), None);
        assert_eq!(pool.get_affinity(Provider::Gemini, "k2").as_deref(), Some("b"));
    }

    #[test]
    fn long_cooldown_threshold_behavior() {
        assert!(!is_long_cooldown(LONG_COOLDOWN_THRESHOLD_MS - 1));
        assert!(is_long_cooldown(LONG_COOLDOWN_THRESHOLD_MS));
        assert!(is_long_cooldown(LONG_COOLDOWN_THRESHOLD_MS + 1));
    }

    #[tokio::test]
    async fn failover_returns_first_success() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                tried.push(rt.to_owned());
                Ok(format!("ok:{rt}"))
            })
            .await
            .unwrap();
        assert_eq!(out, "ok:a");
        assert_eq!(tried, vec!["a"]);
    }

    #[tokio::test]
    async fn failover_cools_rate_limited_account_and_tries_next() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                tried.push(rt.to_owned());
                if rt == "a" {
                    Err(AttemptError::RateLimit(RateLimitError {
                        status: 429,
                        cooldown_ms: 30_000,
                    }))
                } else {
                    Ok(rt.to_owned())
                }
            })
            .await
            .unwrap();
        assert_eq!(out, "b");
        assert_eq!(tried, vec!["a", "b"]);
        assert_eq!(
            pool.available_ids(Provider::Gemini, ModelClass::GeminiNative),
            vec!["b"]
        );
    }

    #[tokio::test]
    async fn failover_all_cooled_throws_rate_limit() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |_| {
                Err::<String, _>(AttemptError::RateLimit(RateLimitError {
                    status: 429,
                    cooldown_ms: 30_000,
                }))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
    }

    #[tokio::test]
    async fn failover_no_accounts_throws_rate_limit() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &[] as &[&str]);
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |_| {
                Ok("x".to_owned())
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
    }

    #[tokio::test]
    async fn failover_generic_error_tries_next() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                tried.push(rt.to_owned());
                if rt == "a" {
                    Err(AttemptError::Other(anyhow::anyhow!("boom")))
                } else {
                    Ok("success-from-b".to_owned())
                }
            })
            .await
            .unwrap();
        assert_eq!(out, "success-from-b");
        assert_eq!(tried, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn failover_deterministic_4xx_fails_fast_without_cooling() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                tried.push(rt.to_owned());
                Err::<String, _>(AttemptError::Upstream(UpstreamError::new(400, "bad request")))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::Upstream(_)));
        assert_eq!(tried, vec!["a"]);
        let mut ids = pool.available_ids(Provider::Gemini, ModelClass::GeminiNative);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn failover_non_account_error_fails_fast() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Codex, &["a", "b"]);
        let mut tried = Vec::new();
        let err = pool
            .with_failover(Provider::Codex, ModelClass::Codex, None, async |rt| {
                tried.push(rt.to_owned());
                Err::<String, _>(AttemptError::NonAccount(NonAccountError::new("socket cap reached")))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::NonAccount(_)));
        assert_eq!(tried, vec!["a"]);
        let mut ids = pool.available_ids(Provider::Codex, ModelClass::Codex);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn failover_auth_error_tries_next() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                tried.push(rt.to_owned());
                if rt == "a" {
                    Err(AttemptError::Upstream(UpstreamError::new(401, "unauthorized")))
                } else {
                    Ok("success-from-b".to_owned())
                }
            })
            .await
            .unwrap();
        assert_eq!(out, "success-from-b");
        assert_eq!(tried, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn failover_spreads_requests_lru() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b", "c"]);
        let mut picked = Vec::new();
        for _ in 0..4 {
            let id = pool
                .with_failover(Provider::Gemini, ModelClass::GeminiNative, None, async |rt| {
                    Ok(rt.to_owned())
                })
                .await
                .unwrap();
            picked.push(id);
        }
        assert_eq!(picked, vec!["a", "b", "c", "a"]);
    }

    #[tokio::test]
    async fn pinned_account_is_tried_when_available() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b", "c"]);
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("c"), async |rt| {
                Ok(rt.to_owned())
            })
            .await
            .unwrap();
        assert_eq!(out, "c");
    }

    #[tokio::test]
    async fn pinned_error_propagates_without_trying_others() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b", "c"]);
        let mut tried = Vec::new();
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("c"), async |rt| {
                tried.push(rt.to_owned());
                Err::<String, _>(AttemptError::RateLimit(RateLimitError {
                    status: 429,
                    cooldown_ms: 30_000,
                }))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
        assert_eq!(tried, vec!["c"]);
    }

    #[tokio::test]
    async fn pinned_cooling_account_throws_without_attempting() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b", "c"]);
        pool.mark_cooldown(Provider::Gemini, "c", ModelClass::GeminiNative, 6000);
        let mut tried = Vec::new();
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("c"), async |rt| {
                tried.push(rt.to_owned());
                Ok(rt.to_owned())
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
        assert_eq!(tried, Vec::<String>::new());
    }

    #[tokio::test]
    async fn pinned_4xx_propagates_and_does_not_cool() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let mut tried = Vec::new();
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("a"), async |rt| {
                tried.push(rt.to_owned());
                Err::<String, _>(AttemptError::Upstream(UpstreamError::new(400, "bad request")))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::Upstream(_)));
        assert_eq!(tried, vec!["a"]);
        let mut ids = pool.available_ids(Provider::Gemini, ModelClass::GeminiNative);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn pinned_rate_limit_cools_for_fresh_sessions() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("a"), async |_| {
                Err::<String, _>(AttemptError::RateLimit(RateLimitError {
                    status: 429,
                    cooldown_ms: 30_000,
                }))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
        assert_eq!(
            pool.available_ids(Provider::Gemini, ModelClass::GeminiNative),
            vec!["b"]
        );
    }

    #[tokio::test]
    async fn pinned_non_account_error_does_not_cool() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Codex, &["a", "b"]);
        let err = pool
            .with_failover(Provider::Codex, ModelClass::Codex, Some("a"), async |_| {
                Err::<String, _>(AttemptError::NonAccount(NonAccountError::new("socket cap reached")))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::NonAccount(_)));
        let mut ids = pool.available_ids(Provider::Codex, ModelClass::Codex);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn pinned_unknown_account_throws_rate_limit() {
        let mut pool = AccountPool::new();
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        let err = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("ghost"), async |rt| {
                Ok(rt.to_owned())
            })
            .await
            .unwrap_err();
        assert!(matches!(err, AttemptError::RateLimit(_)));
    }

    #[tokio::test]
    async fn pinned_long_cooldown_unpins_and_fails_over() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        pool.record_affinity(Provider::Gemini, "key1", "a");
        pool.mark_cooldown(
            Provider::Gemini,
            "a",
            ModelClass::GeminiNative,
            1000 + 12 * 60 * 60 * 1000,
        );

        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("a"), async |rt| {
                tried.push(rt.to_owned());
                Ok(rt.to_owned())
            })
            .await
            .unwrap();
        assert_eq!(out, "b");
        assert_eq!(tried, vec!["b"]);
        assert_eq!(pool.get_affinity(Provider::Gemini, "key1"), None);
        pool.record_account_pin(Provider::Gemini, &["key1"], Some("b"));
        assert_eq!(pool.get_affinity(Provider::Gemini, "key1").as_deref(), Some("b"));
    }

    #[tokio::test]
    async fn pinned_long_rate_limit_unpins_and_fails_over() {
        let mut pool = AccountPool::new();
        pool.set_clock(|| 1000);
        pool.sync_accounts(Provider::Gemini, &["a", "b"]);
        pool.record_affinity(Provider::Gemini, "key1", "a");

        let mut tried = Vec::new();
        let out = pool
            .with_failover(Provider::Gemini, ModelClass::GeminiNative, Some("a"), async |rt| {
                tried.push(rt.to_owned());
                if rt == "a" {
                    Err(AttemptError::RateLimit(RateLimitError {
                        status: 429,
                        cooldown_ms: 12 * 60 * 60 * 1000,
                    }))
                } else {
                    Ok(rt.to_owned())
                }
            })
            .await
            .unwrap();
        assert_eq!(out, "b");
        assert_eq!(tried, vec!["a", "b"]);
        assert_eq!(pool.get_affinity(Provider::Gemini, "key1"), None);
        pool.record_account_pin(Provider::Gemini, &["key1"], Some("b"));
        assert_eq!(pool.get_affinity(Provider::Gemini, "key1").as_deref(), Some("b"));
    }

    fn session_headers(session_id: &str) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-claude-code-session-id", session_id.parse().unwrap());
        headers
    }

    #[test]
    fn preferred_account_resolves_from_header_pin() {
        let mut pool = AccountPool::new();
        let headers = session_headers("session-123");
        let messages = serde_json::json!([
            { "role": "user", "content": "hello" },
            { "role": "assistant", "content": "reply from the previous turn" }
        ]);
        pool.record_affinity(Provider::Gemini, "hdr:session-123", "gemini-account-x");
        let resolved = pool.resolve_preferred_account(Provider::Gemini, &headers, &messages);
        assert_eq!(resolved.account_id.as_deref(), Some("gemini-account-x"));
        assert!(resolved.lookup_keys.contains(&"hdr:session-123".to_owned()));
        assert_eq!(resolved.pin_source, Some(PinSource::Header));
    }

    #[test]
    fn header_only_fresh_request_resolves_nothing() {
        let mut pool = AccountPool::new();
        let headers = session_headers("session-fresh");
        pool.record_affinity(Provider::Gemini, "hdr:session-fresh", "gemini-account-x");
        let resolved = pool.resolve_preferred_account(
            Provider::Gemini,
            &headers,
            &serde_json::json!([{ "role": "user", "content": "new subagent prompt" }]),
        );
        assert_eq!(resolved.account_id, None);
        assert_eq!(resolved.lookup_keys, Vec::<String>::new());
        assert_eq!(resolved.pin_source, None);
    }

    #[test]
    fn claude_session_filters_syncs_and_resolves() {
        use crate::services::credentials::Credentials;
        let mut pool = AccountPool::new();
        let headers = session_headers("session-456");
        let messages = serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "previous answer" }
        ]);
        let creds = Credentials {
            gemini: None,
            claude: Some(serde_json::json!({ "accounts": [
                { "id": "a", "claudeAiOauth": { "accessToken": "x" } },
                { "id": "b", "claudeAiOauth": {} }
            ] })),
            codex: None,
        };
        let session = pool.resolve_claude_session(&headers, &messages, &creds);
        let ids: Vec<String> = session.accounts.iter().map(|a| a.id.clone()).collect();
        assert_eq!(ids, vec!["a"]);
        assert!(
            pool.available_ids(Provider::Claude, ModelClass::Claude)
                .contains(&"a".to_owned())
        );
        assert!(session.lookup_keys.contains(&"hdr:session-456".to_owned()));
    }
}
