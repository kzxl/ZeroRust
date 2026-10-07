//! CANopen CiA 301 Application Layer and Communication Profile.

use crate::can::{CanFrame, CanId};
use zero_core::error::{ZeroError, ZeroResult};

/// CANopen Network Management (NMT) operational state of a slave node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmtState {
    /// Node is initializing and booting up.
    BootUp,
    /// Node is stopped (cannot process SDO or PDO).
    Stopped,
    /// Pre-operational: SDO configuration allowed, PDO transmission disabled.
    PreOperational,
    /// Operational: Full communication allowed including real-time PDO exchange.
    Operational,
}

/// CANopen NMT Control Command sent by the master.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NmtCommand {
    /// Transition node to Operational state (0x01).
    StartNode = 0x01,
    /// Transition node to Stopped state (0x02).
    StopNode = 0x02,
    /// Transition node to Pre-Operational state (0x80).
    EnterPreOperational = 0x80,
    /// Reset node application parameters (0x81).
    ResetNode = 0x81,
    /// Reset node communication stack (0x82).
    ResetCommunication = 0x82,
}

/// CANopen Function Codes for pre-defined connection set (COB-ID = FunctionCode + NodeId).
pub struct CobId;

impl CobId {
    /// NMT Service (0x000).
    pub const NMT: u16 = 0x000;
    /// Synchronization Message SYNC (0x080).
    pub const SYNC: u16 = 0x080;
    /// Emergency Object EMCY (0x080 + NodeId).
    pub const EMCY_BASE: u16 = 0x080;
    /// Transmit PDO 1 (0x180 + NodeId).
    pub const TPDO1_BASE: u16 = 0x180;
    /// Receive PDO 1 (0x200 + NodeId).
    pub const RPDO1_BASE: u16 = 0x200;
    /// Transmit PDO 2 (0x280 + NodeId).
    pub const TPDO2_BASE: u16 = 0x280;
    /// Receive PDO 2 (0x300 + NodeId).
    pub const RPDO2_BASE: u16 = 0x300;
    /// Transmit PDO 3 (0x380 + NodeId).
    pub const TPDO3_BASE: u16 = 0x380;
    /// Receive PDO 3 (0x400 + NodeId).
    pub const RPDO3_BASE: u16 = 0x400;
    /// Transmit PDO 4 (0x480 + NodeId).
    pub const TPDO4_BASE: u16 = 0x480;
    /// Receive PDO 4 (0x500 + NodeId).
    pub const RPDO4_BASE: u16 = 0x500;
    /// Transmit SDO / Slave -> Master (0x580 + NodeId).
    pub const TSDO_BASE: u16 = 0x580;
    /// Receive SDO / Master -> Slave (0x600 + NodeId).
    pub const RSDO_BASE: u16 = 0x600;
    /// Heartbeat / Node Guarding (0x700 + NodeId).
    pub const HEARTBEAT_BASE: u16 = 0x700;
}

/// Builds an NMT Master Command frame.
pub fn build_nmt_command(command: NmtCommand, node_id: u8) -> CanFrame {
    let id = CanId::Standard(CobId::NMT);
    let payload = [command as u8, node_id];
    CanFrame::new(id, &payload).expect("Valid NMT Frame")
}

/// Builds a CANopen SYNC frame (COB-ID 0x080, 0 bytes).
pub fn build_sync_frame() -> CanFrame {
    let id = CanId::Standard(CobId::SYNC);
    CanFrame::new(id, &[]).expect("Valid SYNC Frame")
}

/// SDO Expedited Transfer Helper.
pub struct SdoExpedited;

impl SdoExpedited {
    /// Builds an SDO Download (Write) request frame for up to 4 bytes.
    pub fn build_download_request(
        node_id: u8,
        index: u16,
        subindex: u8,
        value: u32,
        size_bytes: u8,
    ) -> ZeroResult<CanFrame> {
        if size_bytes == 0 || size_bytes > 4 {
            return Err(ZeroError::InvalidArgument);
        }

        // Command Specifier for expedited download:
        // CCS = 1 (bits 7-5 = 001)
        // e = 1 (bit 1 = expedited)
        // s = 1 (bit 0 = size indicated)
        // n = 4 - size_bytes (bits 3-2 = number of unused bytes)
        let n = 4 - size_bytes;
        let cs = 0x20 | (n << 2) | 0x02 | 0x01;

        let index_le = index.to_le_bytes();
        let value_le = value.to_le_bytes();

        let payload = [
            cs,
            index_le[0],
            index_le[1],
            subindex,
            value_le[0],
            value_le[1],
            value_le[2],
            value_le[3],
        ];

        let id = CanId::Standard(CobId::RSDO_BASE + node_id as u16);
        CanFrame::new(id, &payload)
    }

    /// Builds an SDO Upload (Read) request frame.
    pub fn build_upload_request(node_id: u8, index: u16, subindex: u8) -> CanFrame {
        // CCS = 2 (bits 7-5 = 010) -> 0x40
        let index_le = index.to_le_bytes();
        let payload = [
            0x40,
            index_le[0],
            index_le[1],
            subindex,
            0x00,
            0x00,
            0x00,
            0x00,
        ];

        let id = CanId::Standard(CobId::RSDO_BASE + node_id as u16);
        CanFrame::new(id, &payload).expect("Valid SDO Upload frame")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmt_command_frame() {
        let frame = build_nmt_command(NmtCommand::StartNode, 5);
        assert_eq!(frame.id().raw(), 0x000);
        assert_eq!(frame.dlc(), 2);
        assert_eq!(frame.data(), &[0x01, 0x05]);
    }

    #[test]
    fn test_sdo_write_request() {
        // Write 0x1234 (u16 -> 2 bytes) to Index 0x6040, Subindex 0x00 on Node 2
        let frame = SdoExpedited::build_download_request(2, 0x6040, 0x00, 0x1234, 2).unwrap();
        assert_eq!(frame.id().raw(), 0x602);
        assert_eq!(frame.dlc(), 8);
        assert_eq!(frame.data()[0], 0x2B); // 0x20 | (2 << 2) | 0x03 = 0x2B
        assert_eq!(frame.data()[1], 0x40);
        assert_eq!(frame.data()[2], 0x60);
        assert_eq!(frame.data()[3], 0x00);
        assert_eq!(frame.data()[4], 0x34);
        assert_eq!(frame.data()[5], 0x12);
    }
}
