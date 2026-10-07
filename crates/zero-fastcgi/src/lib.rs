//! # ZeroFastCGI
//!
//! Pure Rust, `#![no_std]` capable FastCGI v1.0 binary protocol framing,
//! PHP-FPM Unix domain socket client, and worker pool configuration generator for ZPanl.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod client;
pub mod pool;
pub mod record;

pub use client::FastCgiRequestBuilder;
pub use pool::{PhpPoolConfig, ProcessManagerType};
pub use record::{FcgiHeader, FcgiNameValuePair, FcgiRecordType, FCGI_VERSION_1};
