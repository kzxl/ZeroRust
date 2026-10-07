//! # ZeroCaddy
//!
//! Pure Rust Caddy v2 configuration models, Caddyfile/JSON Vhost builders,
//! and Administration REST API request framing for ZPanl in ZeroRust.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod api;
pub mod builder;
pub mod config;

pub use api::CaddyApiEndpoint;
pub use builder::VhostBuilder;
pub use config::{VhostDescriptor, VhostKind};
