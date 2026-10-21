//! # Upstream Destination & Health State
//!
//! Tracks live destination telemetry using cacheline-aware atomic counters.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use crate::config::DestinationConfig;

/// Live operational state for an upstream destination.
#[derive(Debug)]
pub struct Destination {
    pub address: String,
    pub weight: u32,
    pub role: String,
    pub is_active: bool,
    pub is_healthy: AtomicBool,
    pub in_flight: AtomicUsize,
    pub consecutive_failures: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_latency_us: AtomicU64,
}

impl From<DestinationConfig> for Destination {
    fn from(c: DestinationConfig) -> Self {
        Self {
            address: c.address,
            weight: c.weight.max(1),
            role: c.role,
            is_active: c.is_active,
            is_healthy: AtomicBool::new(true),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }
}

impl Destination {
    #[inline]
    pub fn is_available(&self) -> bool {
        self.is_active && self.is_healthy.load(Ordering::Acquire)
    }

    #[inline]
    pub fn record_start(&self) {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
    }

    #[inline]
    pub fn record_finish(&self, success: bool, latency_us: u64) {
        let prev = self.in_flight.fetch_sub(1, Ordering::AcqRel);
        if prev == 0 {
            // Guard against underflow
            self.in_flight.store(0, Ordering::Release);
        }

        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);

        if success {
            self.consecutive_failures.store(0, Ordering::Release);
            self.is_healthy.store(true, Ordering::Release);
        } else {
            let fails = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
            if fails >= 3 {
                self.is_healthy.store(false, Ordering::Release);
            }
        }
    }

    #[inline]
    pub fn mark_health(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Release);
        if healthy {
            self.consecutive_failures.store(0, Ordering::Release);
        }
    }
}
