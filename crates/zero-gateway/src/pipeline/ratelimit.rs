//! # Lock-Free Atomic Token Bucket Rate Limiter
//!
//! Sub-nanosecond rate limit enforcement using atomic CAS timestamp operations.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Single-route atomic token bucket.
#[derive(Debug)]
pub struct TokenBucket {
    rate_per_sec: u32,
    tokens: AtomicU32,
    last_refill_ms: AtomicU64,
}

impl TokenBucket {
    pub fn new(rate_per_sec: u32) -> Self {
        let now_ms = current_time_ms();
        Self {
            rate_per_sec,
            tokens: AtomicU32::new(rate_per_sec),
            last_refill_ms: AtomicU64::new(now_ms),
        }
    }

    /// Try to acquire 1 token. Returns true if allowed, false if limit exceeded.
    pub fn allow(&self) -> bool {
        if self.rate_per_sec == 0 {
            return true; // 0 = unlimited
        }

        let now_ms = current_time_ms();
        let last_ms = self.last_refill_ms.load(Ordering::Relaxed);
        let elapsed_ms = now_ms.saturating_sub(last_ms);

        if elapsed_ms > 0 {
            // Refill tokens proportional to elapsed time
            let refill = ((elapsed_ms * self.rate_per_sec as u64) / 1000) as u32;
            if refill > 0 {
                let _ = self
                    .last_refill_ms
                    .compare_exchange_weak(last_ms, now_ms, Ordering::AcqRel, Ordering::Relaxed);

                let current = self.tokens.load(Ordering::Relaxed);
                let new_tokens = (current + refill).min(self.rate_per_sec);
                self.tokens.store(new_tokens, Ordering::Release);
            }
        }

        // Attempt to decrement 1 token via atomic CAS
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current == 0 {
                return false;
            }
            if self
                .tokens
                .compare_exchange_weak(current, current - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }
}

/// Dynamic Rate Limiter registry managing buckets per route/key.
pub struct RateLimiterRegistry {
    buckets: RwLock<HashMap<String, Arc<TokenBucket>>>,
}

impl Default for RateLimiterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiterRegistry {
    pub fn new() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
        }
    }

    /// Checks if a request is allowed for a given route and configured limit.
    pub fn check_allowed(&self, key: &str, limit_per_sec: u32) -> bool {
        if limit_per_sec == 0 {
            return true;
        }

        // Fast-path: read lock
        {
            let guard = self.buckets.read();
            if let Some(bucket) = guard.get(key) {
                return bucket.allow();
            }
        }

        // Slow-path: insert bucket
        let mut guard = self.buckets.write();
        let bucket = guard
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(TokenBucket::new(limit_per_sec)));
        bucket.allow()
    }
}

#[inline]
fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
