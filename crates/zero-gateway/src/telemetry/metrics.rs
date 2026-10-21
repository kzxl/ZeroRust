//! # Prometheus & Real-Time Operational Telemetry
//!
//! High-speed metrics accumulation and Prometheus exposition formatting.

use super::histogram::LatencyHistogram;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Snapshot summary for Admin UI dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub total_requests: u64,
    pub success_requests: u64,
    pub client_errors: u64,
    pub server_errors: u64,
    pub active_connections: usize,
    pub success_rate: f64,
    pub latency_p50_ms: f64,
    pub latency_p90_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_avg_ms: f64,
}

/// Global metrics aggregator for the gateway instance.
pub struct MetricsCollector {
    total_requests: AtomicU64,
    success_requests: AtomicU64,
    client_errors: AtomicU64,
    server_errors: AtomicU64,
    active_connections: AtomicUsize,
    histogram: LatencyHistogram,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            success_requests: AtomicU64::new(0),
            client_errors: AtomicU64::new(0),
            server_errors: AtomicU64::new(0),
            active_connections: AtomicUsize::new(0),
            histogram: LatencyHistogram::new(),
        }
    }

    #[inline]
    pub fn conn_open(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn conn_close(&self) {
        let prev = self.active_connections.fetch_sub(1, Ordering::Relaxed);
        if prev == 0 {
            self.active_connections.store(0, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_request(&self, status: u16, latency_us: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if status < 400 {
            self.success_requests.fetch_add(1, Ordering::Relaxed);
        } else if status < 500 {
            self.client_errors.fetch_add(1, Ordering::Relaxed);
        } else {
            self.server_errors.fetch_add(1, Ordering::Relaxed);
        }
        self.histogram.record(latency_us);
    }

    /// Generates a live snapshot for JSON endpoints.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let total = self.total_requests.load(Ordering::Relaxed);
        let success = self.success_requests.load(Ordering::Relaxed);
        let client_err = self.client_errors.load(Ordering::Relaxed);
        let server_err = self.server_errors.load(Ordering::Relaxed);
        let active = self.active_connections.load(Ordering::Relaxed);

        let (p50, p90, p99, avg) = self.histogram.percentiles();
        let rate = if total > 0 {
            (success as f64 / total as f64) * 100.0
        } else {
            100.0
        };

        TelemetrySnapshot {
            total_requests: total,
            success_requests: success,
            client_errors: client_err,
            server_errors: server_err,
            active_connections: active,
            success_rate: (rate * 100.0).round() / 100.0,
            latency_p50_ms: (p50 * 100.0).round() / 100.0,
            latency_p90_ms: (p90 * 100.0).round() / 100.0,
            latency_p99_ms: (p99 * 100.0).round() / 100.0,
            latency_avg_ms: (avg * 100.0).round() / 100.0,
        }
    }

    /// Renders standard Prometheus text exposition format.
    pub fn to_prometheus(&self) -> String {
        let snap = self.snapshot();
        format!(
            "# HELP zgate_requests_total Total HTTP requests processed\n\
             # TYPE zgate_requests_total counter\n\
             zgate_requests_total {}\n\
             # HELP zgate_requests_success Successful HTTP requests (2xx, 3xx)\n\
             # TYPE zgate_requests_success counter\n\
             zgate_requests_success {}\n\
             # HELP zgate_requests_client_errors 4xx Client errors\n\
             # TYPE zgate_requests_client_errors counter\n\
             zgate_requests_client_errors {}\n\
             # HELP zgate_requests_server_errors 5xx Server errors\n\
             # TYPE zgate_requests_server_errors counter\n\
             zgate_requests_server_errors {}\n\
             # HELP zgate_active_connections Current active TCP connections\n\
             # TYPE zgate_active_connections gauge\n\
             zgate_active_connections {}\n\
             # HELP zgate_latency_p99_ms 99th percentile latency in milliseconds\n\
             # TYPE zgate_latency_p99_ms gauge\n\
             zgate_latency_p99_ms {}\n",
            snap.total_requests,
            snap.success_requests,
            snap.client_errors,
            snap.server_errors,
            snap.active_connections,
            snap.latency_p99_ms
        )
    }
}
