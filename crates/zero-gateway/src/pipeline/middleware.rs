//! # Gravitational Middleware Traits & Request Context
//!
//! Complies with Universe Architecture v4.0 Gravitational Pipeline principles.

use crate::error::GatewayError;
use crate::routing::rule::RouteRule;
use std::sync::Arc;

/// Mutable request context flowing through the Gravitational Pipeline.
pub struct RequestContext {
    pub client_ip: String,
    pub method: String,
    pub path: String,
    pub route: Option<Arc<RouteRule>>,
    pub auth_token: Option<String>,
    pub is_websocket: bool,
}

impl RequestContext {
    pub fn new(client_ip: String, method: String, path: String) -> Self {
        Self {
            client_ip,
            method,
            path,
            route: None,
            auth_token: None,
            is_websocket: false,
        }
    }
}

/// Abstract contract for intercepting requests in the Gravitational Pipeline.
pub trait GatewayMiddleware: Send + Sync {
    fn execute(&self, ctx: &mut RequestContext) -> Result<(), GatewayError>;
}
