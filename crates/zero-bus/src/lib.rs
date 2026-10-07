//! # ZeroBus
//!
//! Real-time industrial fieldbus suite for ZeroRust:
//! - CAN 2.0 and CAN-FD abstractions
//! - CANopen CiA 301 Application Layer (NMT, SDO, PDO, SYNC)
//! - CANopen CiA 402 Servo Drive & Motion Control Device Profile

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod can;
pub mod canopen;
pub mod cia402;

pub use can::{CanFrame, CanId, CanInterface};
pub use canopen::{CobId, NmtCommand, NmtState, SdoExpedited};
pub use cia402::{Cia402Drive, ControlWord, DriveState, OperationMode, StatusWord};
