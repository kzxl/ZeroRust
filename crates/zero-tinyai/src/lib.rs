//! # ZeroTinyAI
//!
//! Pure Rust, `#![no_std]` capable Edge Neuromorphic and TinyAI engine:
//! - Int8 symmetric and asymmetric quantization with 32-bit accumulation
//! - Quantized Linear (Dense) and Conv1D layers
//! - Anomaly Detection AutoEncoder for bearing and machine spindle predictive maintenance

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]

pub mod autoencoder;
pub mod conv;
pub mod dense;
pub mod quant;

pub use autoencoder::AnomalyAutoEncoder;
pub use conv::QuantizedConv1D;
pub use dense::QuantizedLinear;
pub use quant::{relu_i8, requantize_i32, QuantParams};
