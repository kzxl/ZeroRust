//! CAN 2.0 and CAN-FD abstractions for industrial communications.

use zero_core::error::{ZeroError, ZeroResult};

/// Represents an 11-bit standard or 29-bit extended CAN identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanId {
    /// 11-bit standard CAN identifier.
    Standard(u16),
    /// 29-bit extended CAN identifier.
    Extended(u32),
}

impl CanId {
    /// Maximum standard CAN identifier (0x7FF = 2047).
    pub const MAX_STANDARD: u16 = 0x7FF;
    /// Maximum extended CAN identifier (0x1FFFFFFF = 536870911).
    pub const MAX_EXTENDED: u32 = 0x1FFF_FFFF;

    /// Creates a new standard 11-bit CAN ID.
    pub const fn new_standard(id: u16) -> Option<Self> {
        if id <= Self::MAX_STANDARD {
            Some(Self::Standard(id))
        } else {
            None
        }
    }

    /// Creates a new extended 29-bit CAN ID.
    pub const fn new_extended(id: u32) -> Option<Self> {
        if id <= Self::MAX_EXTENDED {
            Some(Self::Extended(id))
        } else {
            None
        }
    }

    /// Returns the raw integer value of the identifier.
    pub const fn raw(&self) -> u32 {
        match self {
            Self::Standard(id) => *id as u32,
            Self::Extended(id) => *id,
        }
    }

    /// Returns true if this is an extended 29-bit ID.
    pub const fn is_extended(&self) -> bool {
        matches!(self, Self::Extended(_))
    }
}

/// A CAN frame supporting both classical CAN 2.0 (up to 8 bytes) and CAN-FD (up to 64 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    id: CanId,
    dlc: u8,
    data: [u8; 64],
    is_fd: bool,
    is_rtr: bool,
}

impl CanFrame {
    /// Creates a new standard CAN 2.0 data frame (up to 8 bytes).
    pub fn new(id: CanId, data_slice: &[u8]) -> ZeroResult<Self> {
        if data_slice.len() > 8 {
            return Err(ZeroError::InvalidArgument);
        }
        let mut data = [0u8; 64];
        data[..data_slice.len()].copy_from_slice(data_slice);
        Ok(Self {
            id,
            dlc: data_slice.len() as u8,
            data,
            is_fd: false,
            is_rtr: false,
        })
    }

    /// Creates a new CAN-FD data frame (up to 64 bytes).
    pub fn new_fd(id: CanId, data_slice: &[u8]) -> ZeroResult<Self> {
        if data_slice.len() > 64 {
            return Err(ZeroError::InvalidArgument);
        }
        let mut data = [0u8; 64];
        data[..data_slice.len()].copy_from_slice(data_slice);
        Ok(Self {
            id,
            dlc: data_slice.len() as u8,
            data,
            is_fd: true,
            is_rtr: false,
        })
    }

    /// Returns the CAN identifier.
    #[inline]
    pub const fn id(&self) -> CanId {
        self.id
    }

    /// Returns the data length code (DLC) / payload length.
    #[inline]
    pub const fn dlc(&self) -> u8 {
        self.dlc
    }

    /// Returns a slice of the active payload data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data[..self.dlc as usize]
    }

    /// Returns true if this is a CAN-FD frame.
    #[inline]
    pub const fn is_fd(&self) -> bool {
        self.is_fd
    }

    /// Returns true if this is a Remote Transmission Request (RTR).
    #[inline]
    pub const fn is_rtr(&self) -> bool {
        self.is_rtr
    }
}

/// Interface trait for CAN hardware adapters and virtual buses.
pub trait CanInterface {
    /// Transmits a CAN frame onto the bus.
    fn transmit(&mut self, frame: &CanFrame) -> ZeroResult<()>;
    /// Receives an incoming CAN frame if available.
    fn receive(&mut self) -> ZeroResult<Option<CanFrame>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_frame_creation() {
        let id = CanId::new_standard(0x123).unwrap();
        let payload = [1, 2, 3, 4];
        let frame = CanFrame::new(id, &payload).unwrap();

        assert_eq!(frame.id().raw(), 0x123);
        assert_eq!(frame.dlc(), 4);
        assert_eq!(frame.data(), &[1, 2, 3, 4]);
        assert!(!frame.is_fd());
    }
}
