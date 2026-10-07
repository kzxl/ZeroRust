//! # ZeroSys
//!
//! Native Linux system metrics (/proc zero-alloc parser), Systemd service management,
//! and POSIX security validation for ZPanl in the ZeroRust ecosystem.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod proc;
pub mod service;
pub mod user;

pub use proc::{CpuStats, MemStats, NetInterfaceStats};
pub use service::{ServiceAction, ServiceState, SystemdManager};
pub use user::PosixPermissions;
