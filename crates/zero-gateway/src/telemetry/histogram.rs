//! # Sub-Microsecond Lock-Free Latency Histogram
//!
//! Tracks request duration buckets and percentile approximations (P50, P90, P99) with zero allocation.

use std::sync::atomic::{AtomicU64, Ordering};

const BUCKET_COUNT: usize = 10;
const BUCKET_LIMITS_US: [u64; BUCKET_COUNT] = [
    500,       // < 0.5 ms
    1_000,     // < 1 ms
    2_500,     // < 2.5 ms
    5_000,     // < 5 ms
    10_000,    // < 10 ms
    25_000,    // < 25 ms
    50_000,    // < 50 ms
    100_000,   // < 100 ms
    250_000,   // < 250 ms
    u64::MAX,  // >= 250 ms
];

/// Lock-free latency histogram tracking request timing distributions.
pub struct LatencyHistogram {
    buckets: [AtomicU64; BUCKET_COUNT],
    total_count: AtomicU64,
    sum_us: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0),
            ],
            total_count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
        }
    }

    /// Records a latency sample in microseconds.
    pub fn record(&self, latency_us: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(latency_us, Ordering::Relaxed);

        for (i, &limit) in BUCKET_LIMITS_US.iter().enumerate() {
            if latency_us <= limit {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Calculates estimated percentiles in milliseconds: (p50, p90, p99, avg).
    pub fn percentiles(&self) -> (f64, f64, f64, f64) {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let sum = self.sum_us.load(Ordering::Relaxed);
        let avg = (sum as f64 / total as f64) / 1000.0;

        let p50_target = (total as f64 * 0.50) as u64;
        let p90_target = (total as f64 * 0.90) as u64;
        let p99_target = (total as f64 * 0.99) as u64;

        let mut cum: u64 = 0;
        let mut p50_us = 0.0;
        let mut p90_us = 0.0;
        let mut p99_us = 0.0;

        for (i, b) in self.buckets.iter().enumerate() {
            cum += b.load(Ordering::Relaxed);
            let limit = BUCKET_LIMITS_US[i] as f64 / 1000.0;

            if p50_us == 0.0 && cum >= p50_target {
                p50_us = limit;
            }
            if p90_us == 0.0 && cum >= p90_target {
                p90_us = limit;
            }
            if p99_us == 0.0 && cum >= p99_target {
                p99_us = limit;
            }
        }

        (p50_us, p90_us, p99_us, avg)
    }
}
