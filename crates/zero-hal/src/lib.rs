//! # ZeroHAL
//!
//! Industrial Hardware Abstraction Layer for ZeroRust:
//! - Digital Input/Output abstractions with software Debouncing
//! - 4x Quadrature Encoder decoding with Index Z latching
//! - Fractional DDA Step/Dir pulse train generator for stepper and servo drives

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod encoder;
pub mod io;
pub mod stepdir;

pub use encoder::{Direction, QuadratureDecoder};
pub use io::{Debouncer, DigitalInput, DigitalOutput, MockPin};
pub use stepdir::StepDirGenerator;
