//! # Gateway Configuration Schema
//!
//! Strongly-typed, serializable configuration models for routes, clusters,
//! security parameters, and runtime hardware tuning.

use serde::{Deserialize, Serialize};

/// Master configuration for the ZGate engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Server listener and worker settings.
    pub server: ServerConfig,
    /// Static or dynamically loaded route definitions.
    pub routes: Vec<RouteConfig>,
    /// Upstream target clusters.
    pub clusters: Vec<ClusterConfig>,
}

/// Listener and operational limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address for proxy traffic (e.g. "0.0.0.0").
    pub bind_address: String,
    /// Gateway listener port (e.g. 8887).
    pub port: u16,
    /// Admin API and metrics listener port (e.g. 8888).
    pub admin_port: u16,
    /// Number of worker threads (0 for auto physical core detection).
    pub worker_threads: usize,
    /// Max active TCP connections before rejecting with 503.
    pub max_connections: usize,
    /// Pre-allocated buffer size in KB (default: 32).
    pub buffer_size_kb: usize,
    /// Connection timeout to upstream in milliseconds.
    pub upstream_timeout_ms: u64,
    /// Shared secret for HMAC-SHA256 JWT validation.
    pub jwt_secret: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8887,
            admin_port: 8888,
            worker_threads: 0,
            max_connections: 50_000,
            buffer_size_kb: 32,
            upstream_timeout_ms: 10_000,
            jwt_secret: "zgate-default-secret-key-32chars-min!!".to_string(),
        }
    }
}

/// Dynamic route matching rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    /// Unique identifier for this route.
    pub route_id: String,
    /// Path prefix to match (e.g. "/api/v1/orders").
    pub match_path: String,
    /// Target cluster identifier.
    pub cluster_id: String,
    /// Strip matched prefix before forwarding to upstream.
    pub strip_prefix: bool,
    /// Rate limit per second (0 for unlimited).
    pub rate_limit_per_second: u32,
    /// Whether JWT authentication is mandatory.
    pub auth_required: bool,
    /// Whether this route is currently active.
    pub is_active: bool,
}

/// Upstream cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Unique identifier for this cluster.
    pub cluster_id: String,
    /// Human-readable cluster name.
    pub name: String,
    /// Load balancing algorithm: "RoundRobin", "LeastConnections", "Random".
    pub load_balancing_policy: String,
    /// List of upstream target endpoints.
    pub destinations: Vec<DestinationConfig>,
    /// Path for active health check probes (e.g. "/health").
    pub health_check_path: String,
    /// Probing interval in seconds.
    pub health_check_interval_sec: u64,
    /// Whether this cluster is active.
    pub is_active: bool,
}

/// Individual upstream destination node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinationConfig {
    /// Upstream socket address (e.g. "127.0.0.1:5001" or "http://backend.internal:8080").
    pub address: String,
    /// Traffic distribution weight (1-100).
    pub weight: u32,
    /// Operational role: "Active" or "Standby".
    pub role: String,
    /// Whether this node is active.
    pub is_active: bool,
}
