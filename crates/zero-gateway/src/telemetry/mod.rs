//! # Telemetry & Monitoring Subsystem
//!
//! Real-time latency tracking, Prometheus metrics exporter, and Admin health stats.

pub mod histogram;
pub mod metrics;

pub use histogram::LatencyHistogram;
pub use metrics::{MetricsCollector, TelemetrySnapshot};
