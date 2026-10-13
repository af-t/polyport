use crate::types::{ModelClass, Provider};
use std::collections::HashMap;

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
}
