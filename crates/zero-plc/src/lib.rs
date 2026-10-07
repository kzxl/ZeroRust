//! # ZeroPLC
//!
//! Pure Rust, `#![no_std]` capable deterministic Soft-PLC Runtime:
//! - IEC 61131-3 Process Image Memory Model (`%I`, `%Q`, `%M`)
//! - Stack-based Virtual Machine bytecode interpreter
//! - Standard IEC 61131-3 Function Blocks: `TON`, `TOF`, `TP`, `CTU`, `CTD`, `PID_Compact`
//! - Zero-downtime hot-reload preserving live memory states and axis positions

#![cfg_attr(not(feature = "std"), no_std)]

pub mod fb;
pub mod instruction;
pub mod memory;
pub mod vm;

pub use fb::{Ctd, Ctu, PidCompact, Tof, Ton, Tp};
pub use instruction::Instruction;
pub use memory::{PlcAddress, PlcArea, PlcMemory};
pub use vm::{FunctionBlockInstance, PlcCycleStats, PlcVm};
