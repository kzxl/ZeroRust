//! # ZeroCore
//!
//! Deterministic `#![no_std]` foundation primitives for the ZeroRust ecosystem:
//! - Lock-free SPSC Ring Buffers
//! - Zero-copy binary span readers and writers
//! - Deterministic fixed-point arithmetic (`Q16_16`)
//! - Unified error model

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod error;
pub mod fixed_point;
pub mod ring_buffer;
pub mod span;

pub use error::{ZeroError, ZeroResult};
pub use fixed_point::Q16_16;
pub use ring_buffer::SpscRingBuffer;
pub use span::{ByteReader, ByteWriter};
