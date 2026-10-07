//! # ZeroIPC
//!
//! Ultra-low-latency, zero-copy Shared Memory IPC bridge for ZeroRust:
//! - Cacheline-aligned (64-byte) cross-language Shared Memory Header compatible with C# `MemoryMarshal`
//! - Lock-free multi-slot ring buffer using monotonic atomic sequence numbers
//! - Standard structured payloads for real-time motion setpoints and machine telemetry

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod layout;
pub mod payloads;
pub mod ring;

pub use layout::{ShmHeader, IPC_MAGIC, IPC_VERSION};
pub use payloads::{MotionCommandPayload, TelemetryPayload};
pub use ring::ShmRingBuffer;
