//! # ZeroGateway (`zero-gateway`)
//!
//! Sovereign, ultra-high-throughput, sub-millisecond API Gateway & L7 reverse proxy for ZeroUniverse.
//!
//! - **Prefix Radix Trie Router**: O(K) lookup complexity
//! - **Multi-Strategy Atomic Load Balancers**: Round-Robin, Least-Connections, Random
//! - **Zero-Allocation Hot Path**: Cacheline-aligned pre-allocated buffer pool
//! - **Active Upstream Health Probing**: Autonomous circuit breaker
//! - **Lock-Free Rate Limiting**: Atomic Token Bucket limiter
//! - **Zero-Panic Invariants**: Explicit typed error modeling

pub mod auth;
pub mod cluster;
pub mod config;
pub mod error;
pub mod pipeline;
pub mod proxy;
pub mod routing;
pub mod server;
pub mod telemetry;

pub use config::{ClusterConfig, DestinationConfig, GatewayConfig, RouteConfig, ServerConfig};
pub use error::GatewayError;
pub use server::GatewayServer;
