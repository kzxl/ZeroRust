//! # Route Rule & Path Rewriting Logic
//!
//! Strongly-typed routing rules for high-speed prefix matching and upstream targeting.

use crate::config::RouteConfig;

/// Compiled route rule ready for hot-path evaluation.
#[derive(Debug, Clone)]
pub struct RouteRule {
    pub route_id: String,
    pub match_path: String,
    pub cluster_id: String,
    pub strip_prefix: bool,
    pub rate_limit_per_second: u32,
    pub auth_required: bool,
    pub is_active: bool,
}

impl From<RouteConfig> for RouteRule {
    fn from(c: RouteConfig) -> Self {
        // Normalize match_path: ensure leading slash, strip trailing slash unless root
        let mut path = c.match_path.trim().to_string();
        if !path.starts_with('/') {
            path.insert(0, '/');
        }
        if path.len() > 1 && path.ends_with('/') {
            path.pop();
        }

        Self {
            route_id: c.route_id,
            match_path: path,
            cluster_id: c.cluster_id,
            strip_prefix: c.strip_prefix,
            rate_limit_per_second: c.rate_limit_per_second,
            auth_required: c.auth_required,
            is_active: c.is_active,
        }
    }
}

impl RouteRule {
    /// Rewrites the incoming request path based on the route configuration.
    #[inline]
    pub fn rewrite_path<'a>(&self, req_path: &'a str) -> &'a str {
        if !self.strip_prefix {
            return req_path;
        }

        if req_path.starts_with(&self.match_path) {
            let remainder = &req_path[self.match_path.len()..];
            if remainder.is_empty() {
                "/"
            } else if remainder.starts_with('/') {
                remainder
            } else {
                req_path
            }
        } else {
            req_path
        }
    }
}
