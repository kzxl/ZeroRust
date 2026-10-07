//! Modbus Protocol Data Unit (PDU) structures and definitions.

use zero_core::error::{ZeroError, ZeroResult};

/// Standard Modbus Function Codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModbusFunction {
    /// Read Coils (0x01).
    ReadCoils = 0x01,
    /// Read Discrete Inputs (0x02).
    ReadDiscreteInputs = 0x02,
    /// Read Holding Registers (0x03).
    ReadHoldingRegisters = 0x03,
    /// Read Input Registers (0x04).
    ReadInputRegisters = 0x04,
    /// Write Single Coil (0x05).
    WriteSingleCoil = 0x05,
    /// Write Single Register (0x06).
    WriteSingleRegister = 0x06,
    /// Write Multiple Coils (0x0F).
    WriteMultipleCoils = 0x0F,
    /// Write Multiple Registers (0x10).
    WriteMultipleRegisters = 0x10,
}

impl TryFrom<u8> for ModbusFunction {
    type Error = ZeroError;

    fn try_from(code: u8) -> Result<Self, Self::Error> {
        match code {
            0x01 => Ok(Self::ReadCoils),
            0x02 => Ok(Self::ReadDiscreteInputs),
            0x03 => Ok(Self::ReadHoldingRegisters),
            0x04 => Ok(Self::ReadInputRegisters),
            0x05 => Ok(Self::WriteSingleCoil),
            0x06 => Ok(Self::WriteSingleRegister),
            0x0F => Ok(Self::WriteMultipleCoils),
            0x10 => Ok(Self::WriteMultipleRegisters),
            _ => Err(ZeroError::Unsupported),
        }
    }
}

/// Standard Modbus Exception Codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModbusException {
    /// Function code received is not an allowable action for the server (0x01).
    IllegalFunction = 0x01,
    /// Address received is not an allowable address for the server (0x02).
    IllegalDataAddress = 0x02,
    /// Value contained in query data field is not allowable (0x03).
    IllegalDataValue = 0x03,
    /// Unrecoverable error occurred while server was attempting to perform requested action (0x04).
    ServerDeviceFailure = 0x04,
    /// Server has accepted request and is processing it, but long duration is required (0x05).
    Acknowledge = 0x05,
    /// Server is engaged in processing a long-duration program command (0x06).
    ServerDeviceBusy = 0x06,
}

/// A parsed Modbus request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModbusRequest {
    /// Read Holding Registers (0x03): `address`, `quantity`.
    ReadHoldingRegisters {
        /// Starting address.
        address: u16,
        /// Quantity of registers to read.
        quantity: u16,
    },
    /// Read Input Registers (0x04): `address`, `quantity`.
    ReadInputRegisters {
        /// Starting address.
        address: u16,
        /// Quantity of registers to read.
        quantity: u16,
    },
    /// Write Single Register (0x06): `address`, `value`.
    WriteSingleRegister {
        /// Register address.
        address: u16,
        /// Register value.
        value: u16,
    },
    /// Write Single Coil (0x05): `address`, `state`.
    WriteSingleCoil {
        /// Coil address.
        address: u16,
        /// Coil state (true = ON / 0xFF00, false = OFF / 0x0000).
        state: bool,
    },
}

impl ModbusRequest {
    /// Parses a Modbus PDU byte slice.
    pub fn parse_pdu(pdu: &[u8]) -> ZeroResult<Self> {
        if pdu.is_empty() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let function = ModbusFunction::try_from(pdu[0])?;
        match function {
            ModbusFunction::ReadHoldingRegisters => {
                if pdu.len() < 5 {
                    return Err(ZeroError::UnexpectedEndOfBuffer);
                }
                let address = u16::from_be_bytes([pdu[1], pdu[2]]);
                let quantity = u16::from_be_bytes([pdu[3], pdu[4]]);
                Ok(Self::ReadHoldingRegisters { address, quantity })
            }
            ModbusFunction::ReadInputRegisters => {
                if pdu.len() < 5 {
                    return Err(ZeroError::UnexpectedEndOfBuffer);
                }
                let address = u16::from_be_bytes([pdu[1], pdu[2]]);
                let quantity = u16::from_be_bytes([pdu[3], pdu[4]]);
                Ok(Self::ReadInputRegisters { address, quantity })
            }
            ModbusFunction::WriteSingleRegister => {
                if pdu.len() < 5 {
                    return Err(ZeroError::UnexpectedEndOfBuffer);
                }
                let address = u16::from_be_bytes([pdu[1], pdu[2]]);
                let value = u16::from_be_bytes([pdu[3], pdu[4]]);
                Ok(Self::WriteSingleRegister { address, value })
            }
            ModbusFunction::WriteSingleCoil => {
                if pdu.len() < 5 {
                    return Err(ZeroError::UnexpectedEndOfBuffer);
                }
                let address = u16::from_be_bytes([pdu[1], pdu[2]]);
                let raw_val = u16::from_be_bytes([pdu[3], pdu[4]]);
                let state = match raw_val {
                    0xFF00 => true,
                    0x0000 => false,
                    _ => return Err(ZeroError::InvalidArgument),
                };
                Ok(Self::WriteSingleCoil { address, state })
            }
            _ => Err(ZeroError::Unsupported),
        }
    }
}
