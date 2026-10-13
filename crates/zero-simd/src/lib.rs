//! # ZeroSIMD
//!
//! Deterministic hardware-intrinsic SIMD microkernels, fast math, activations,
//! and quantization primitives for ZeroUniverse (no_std capable).

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]

pub mod fast_math;
pub mod microkernel;
pub mod norm;
pub mod quant;
pub mod rope;

pub use fast_math::*;
pub use microkernel::*;
pub use norm::*;
pub use quant::*;
pub use rope::*;
