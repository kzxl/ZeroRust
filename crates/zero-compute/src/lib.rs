//! # ZeroCompute
//!
//! Sovereign, high-performance compute substrate, cache-aware schedulers,
//! and hardware device abstraction runtime for ZeroUniverse.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments)]

pub mod cpu;
pub mod device;
pub mod gpu;
pub mod scheduler;
pub mod shaders;

pub use cpu::*;
pub use device::*;
pub use gpu::*;
pub use scheduler::*;
pub use shaders::*;


