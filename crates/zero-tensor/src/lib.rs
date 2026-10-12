//! # ZeroTensor
//!
//! Sovereign, high-performance native compute substrate & microkernel acceleration for ZeroUniverse:
//! - Cache-tiled GEMM with AVX2/FMA register microkernels (4x16 tile) and multi-threaded row partitioning
//! - 2D Block-Tiled FlashAttention-2 with online softmax and SIMD dot products
//! - Streaming on-the-fly Quantized GEMV (Q4_0, Q8_0)
//! - ABI-stable C-FFI for ZeroPlatform (.NET C# P/Invoke) integration

#![cfg_attr(not(feature = "std"), no_std)]

pub mod gemm;
pub mod attention;
pub mod quant;
pub mod ffi;

pub use gemm::gemm_f32;
pub use attention::flash_attention_f32;
pub use quant::{gemv_q4_0, gemv_q8_0};
pub use ffi::*;
