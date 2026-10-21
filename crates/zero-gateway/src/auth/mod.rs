//! # Authentication & Authorization Subsystem
//!
//! High-speed JWT validation with L1 caching and role-based access checks.

pub mod cache;
pub mod jwt;

use cache::TokenCache;
use jwt::{validate_jwt, JwtClaims};
use std::sync::Arc;
use crate::error::GatewayError;

/// Coordinator for verifying client authentication tokens.
#[derive(Clone)]
pub struct AuthManager {
    secret: Arc<Vec<u8>>,
    cache: TokenCache,
}

impl AuthManager {
    pub fn new(secret: &str) -> Self {
        Self {
            secret: Arc::new(secret.as_bytes().to_vec()),
            cache: TokenCache::new(),
        }
    }

    /// Verifies the provided token string (Bearer <token>).
    pub fn verify_token(&self, token_str: &str) -> Result<JwtClaims, GatewayError> {
        let clean = token_str.trim_start_matches("Bearer ").trim();
        if clean.is_empty() {
            return Err(GatewayError::AuthTokenMissing);
        }

        // Fast-path: check L1 cache
        if let Some(claims) = self.cache.get(clean) {
            return Ok(claims);
        }

        // Slow-path: cryptographic validation
        let claims = validate_jwt(clean, &self.secret)?;

        // Cache result for 60 seconds
        self.cache.insert(clean.to_string(), claims.clone(), 60);

        Ok(claims)
    }
}
