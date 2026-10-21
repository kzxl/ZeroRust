//! # Strongly-Typed Gateway Error Invariants
//!
//! Complies with AgentOption Rust Safety Standards (R_RUST, R_CORE).
//! Strictly typed enum deriving Debug, Clone, PartialEq, Eq. Zero runtime panics.

use std::fmt;

/// Sovereign error domain for the ZGate proxy engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    /// No route matched the incoming HTTP request path.
    NoRouteMatched { path: String },
    /// No healthy upstream destination is available in the target cluster.
    NoHealthyUpstream { cluster_id: String },
    /// Rate limit exceeded for the client or route.
    RateLimitExceeded { limit: u32 },
    /// Authentication token is missing from the request header.
    AuthTokenMissing,
    /// Authentication token is invalid or expired.
    AuthTokenInvalid { reason: String },
    /// Failed to connect to the target upstream destination.
    UpstreamConnectionFailed { address: String, details: String },
    /// Upstream response timed out before completion.
    UpstreamTimeout { timeout_ms: u64 },
    /// Malformed HTTP request framing or headers.
    MalformedRequest { reason: String },
    /// Upstream returned an invalid HTTP response format.
    MalformedUpstreamResponse { reason: String },
    /// Internal socket or I/O error occurred.
    IoFailure { details: String },
    /// Storage or persistence failure.
    StorageFailure { details: String },
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRouteMatched { path } => {
                write!(f, "No route matched request path: '{path}'")
            }
            Self::NoHealthyUpstream { cluster_id } => {
                write!(f, "No healthy upstream destination for cluster: '{cluster_id}'")
            }
            Self::RateLimitExceeded { limit } => {
                write!(f, "Rate limit exceeded ({limit} req/s)")
            }
            Self::AuthTokenMissing => {
                write!(f, "Authentication token missing from Authorization header")
            }
            Self::AuthTokenInvalid { reason } => {
                write!(f, "Authentication token invalid: {reason}")
            }
            Self::UpstreamConnectionFailed { address, details } => {
                write!(f, "Failed to connect to upstream '{address}': {details}")
            }
            Self::UpstreamTimeout { timeout_ms } => {
                write!(f, "Upstream request timed out after {timeout_ms} ms")
            }
            Self::MalformedRequest { reason } => {
                write!(f, "Malformed HTTP request: {reason}")
            }
            Self::MalformedUpstreamResponse { reason } => {
                write!(f, "Malformed upstream HTTP response: {reason}")
            }
            Self::IoFailure { details } => {
                write!(f, "I/O failure: {details}")
            }
            Self::StorageFailure { details } => {
                write!(f, "Storage failure: {details}")
            }
        }
    }
}

impl std::error::Error for GatewayError {}

impl From<std::io::Error> for GatewayError {
    fn from(err: std::io::Error) -> Self {
        Self::IoFailure {
            details: err.to_string(),
        }
    }
}
