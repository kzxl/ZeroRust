//! # ZeroEtherCAT
//!
//! Pure Rust, `#![no_std]` capable Industrial Real-Time EtherCAT Master engine.
//!
//! Provides zero-allocation framing, datagram serialization, EtherCAT State Machine (ESM),
//! CANopen over EtherCAT (CoE) SDO mailbox processing, and Distributed Clocks (DC) time synchronization.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod coe;
pub mod dc;
pub mod esm;
pub mod frame;
pub mod master;

pub use coe::{CoeSdo, CoeService, MailboxHeader, MailboxType};
pub use dc::{DcConfig, DcEngine, PortTimestamps};
pub use esm::{is_valid_esm_transition, AlStatus, AlStatusCode, EscRegister, EsmState};
pub use frame::{EthercatCmd, EthercatDatagram, EthercatFrame, ETHERTYPE_ETHERCAT};
pub use master::{EthercatMaster, SlaveInfo};
