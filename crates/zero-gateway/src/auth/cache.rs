//! # L1 Token Verification Cache
//!
//! Sub-nanosecond cached JWT verification eliminating redundant HMAC cycles on repeated requests.

use super::jwt::JwtClaims;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct CachedToken {
    claims: JwtClaims,
    cached_until_sec: u64,
}

/// In-memory thread-safe cache for validated JWT tokens.
#[derive(Clone)]
pub struct TokenCache {
    entries: Arc<RwLock<HashMap<String, CachedToken>>>,
}

impl Default for TokenCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCache {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Attempts to retrieve valid cached claims for the given token.
    pub fn get(&self, token: &str) -> Option<JwtClaims> {
        let now = current_time_sec();
        let guard = self.entries.read();
        if let Some(entry) = guard.get(token) {
            if entry.cached_until_sec > now {
                return Some(entry.claims.clone());
            }
        }
        None
    }

    /// Stores validated claims in cache with a TTL (e.g. 60 seconds or until token expiration).
    pub fn insert(&self, token: String, claims: JwtClaims, ttl_sec: u64) {
        let now = current_time_sec();
        let token_exp = if claims.exp > 0 { claims.exp } else { now + ttl_sec };
        let cached_until = (now + ttl_sec).min(token_exp);

        let mut guard = self.entries.write();
        // Evict expired entries if cache grows large
        if guard.len() > 5000 {
            guard.retain(|_, v| v.cached_until_sec > now);
        }

        guard.insert(
            token,
            CachedToken {
                claims,
                cached_until_sec: cached_until,
            },
        );
    }
}

#[inline]
fn current_time_sec() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
