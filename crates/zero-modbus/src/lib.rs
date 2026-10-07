//! # ZeroModbus
//!
//! High-performance, zero-allocation Modbus protocol engine for ZeroRust:
//! - CRC16-Modbus computation
//! - Modbus PDU parsing (Holding Registers, Input Registers, Coils)
//! - Modbus RTU (Serial/RS485) Frame Encoder & Decoder
//! - Modbus TCP (MBAP Header) Frame Parser

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod crc;
pub mod pdu;
pub mod rtu;
pub mod tcp;

pub use crc::calculate_crc16;
pub use pdu::{ModbusException, ModbusFunction, ModbusRequest};
pub use rtu::ModbusRtu;
pub use tcp::{MbapHeader, ModbusTcpFrame};
