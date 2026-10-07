//! # ZeroDSP
//!
//! Real-time Digital Signal Processing suite for ZeroRust:
//! - 2nd-Order IIR Butterworth Low-Pass filter
//! - Moving Average filter with compile-time window size
//! - 1D Linear Kalman Filter for optimal sensor fusion
//! - Radix-2 Cooley-Tukey Fast Fourier Transform (FFT) for vibration spectrum analysis

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod fft;
pub mod filter;
pub mod kalman;

pub use fft::{Complex64, Radix2Fft};
pub use filter::{ButterworthLowPass, MovingAverage};
pub use kalman::KalmanFilter1D;
