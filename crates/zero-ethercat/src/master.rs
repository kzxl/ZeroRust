//! EtherCAT Master orchestrator: slave discovery, address configuration, ESM state coordination, and cyclic PDO exchange.

use crate::esm::{AlStatus, EscRegister, EsmState};
use crate::frame::{EthercatCmd, EthercatDatagram};
use zero_core::error::{ZeroError, ZeroResult};

/// Information tracked for each detected EtherCAT slave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlaveInfo {
    /// Auto-increment address assigned during initial enumeration (0, -1, -2...).
    pub auto_inc_address: u16,
    /// Configured station address assigned by master (e.g. 0x1001, 0x1002...).
    pub station_address: u16,
    /// Current ESM state of the slave.
    pub current_state: EsmState,
    /// Expected WKC increment for this slave.
    pub expected_wkc: u16,
}

impl Default for SlaveInfo {
    fn default() -> Self {
        Self {
            auto_inc_address: 0,
            station_address: 0,
            current_state: EsmState::Init,
            expected_wkc: 0,
        }
    }
}

/// EtherCAT Master runtime orchestrator with fixed-capacity slave table (zero-alloc `#![no_std]`).
#[derive(Debug)]
pub struct EthercatMaster<const MAX_SLAVES: usize> {
    /// Array of discovered and managed slaves.
    pub slaves: [SlaveInfo; MAX_SLAVES],
    /// Number of active slaves currently managed.
    pub slave_count: usize,
    /// Master target state for the whole network.
    pub network_state: EsmState,
    /// Rolling datagram sequence index.
    seq_index: u8,
}

impl<const MAX_SLAVES: usize> EthercatMaster<MAX_SLAVES> {
    /// Creates a new EtherCAT Master instance.
    pub fn new() -> Self {
        Self {
            slaves: [SlaveInfo::default(); MAX_SLAVES],
            slave_count: 0,
            network_state: EsmState::Init,
            seq_index: 0,
        }
    }

    /// Allocates the next rolling sequence index.
    pub fn next_index(&mut self) -> u8 {
        let idx = self.seq_index;
        self.seq_index = self.seq_index.wrapping_add(1);
        idx
    }

    /// Builds a Broadcast Read (BRD) datagram to query AL Status across all slaves in the ring.
    ///
    /// The Working Counter (WKC) returned by the ring will equal the number of responsive slaves.
    pub fn build_scan_datagram<'a>(
        &mut self,
        dummy_payload: &'a mut [u8; 2],
    ) -> EthercatDatagram<'a> {
        let index = self.next_index();
        // Clear dummy payload
        dummy_payload.fill(0);
        EthercatDatagram {
            cmd: EthercatCmd::Brd,
            index,
            address: EscRegister::AL_STATUS as u32,
            data: dummy_payload,
            wkc: 0,
        }
    }

    /// Registers a newly discovered slave into the master's table.
    pub fn register_slave(
        &mut self,
        auto_inc_address: u16,
        station_address: u16,
    ) -> ZeroResult<usize> {
        if self.slave_count >= MAX_SLAVES {
            return Err(ZeroError::BufferOverflow);
        }

        let idx = self.slave_count;
        self.slaves[idx] = SlaveInfo {
            auto_inc_address,
            station_address,
            current_state: EsmState::Init,
            expected_wkc: 3, // Typical LRW contribution (In + Out)
        };
        self.slave_count += 1;
        Ok(idx)
    }

    /// Builds an Auto-Increment Write (APWR) datagram to assign a station address to a slave.
    pub fn build_set_station_address_datagram<'a>(
        &mut self,
        slave_idx: usize,
        station_addr: u16,
        payload_buf: &'a mut [u8; 2],
    ) -> ZeroResult<EthercatDatagram<'a>> {
        if slave_idx >= self.slave_count {
            return Err(ZeroError::InvalidArgument);
        }

        let auto_inc_addr = self.slaves[slave_idx].auto_inc_address;
        let index = self.next_index();

        *payload_buf = station_addr.to_le_bytes();

        // 32-bit address: high 16 bits = auto-inc address, low 16 bits = register
        let address = ((auto_inc_addr as u32) << 16) | (EscRegister::STATION_ADDRESS as u32);

        Ok(EthercatDatagram {
            cmd: EthercatCmd::Apwr,
            index,
            address,
            data: payload_buf,
            wkc: 0,
        })
    }

    /// Builds a Configured Address Write (FPWR) datagram to command an ESM state transition for a slave.
    pub fn build_state_transition_datagram<'a>(
        &mut self,
        slave_idx: usize,
        target_state: EsmState,
        payload_buf: &'a mut [u8; 2],
    ) -> ZeroResult<EthercatDatagram<'a>> {
        if slave_idx >= self.slave_count {
            return Err(ZeroError::InvalidArgument);
        }

        let station_addr = self.slaves[slave_idx].station_address;
        let index = self.next_index();

        let control_word = AlStatus::build_control_word(target_state, false);
        *payload_buf = control_word.to_le_bytes();

        let address = ((station_addr as u32) << 16) | (EscRegister::AL_CONTROL as u32);

        Ok(EthercatDatagram {
            cmd: EthercatCmd::Fpwr,
            index,
            address,
            data: payload_buf,
            wkc: 0,
        })
    }

    /// Builds a cyclic Logical Read Write (LRW) datagram for exchange of real-time Process Data (PDOs).
    pub fn build_cyclic_lrw_datagram<'a>(
        &mut self,
        logical_address: u32,
        process_data: &'a mut [u8],
    ) -> EthercatDatagram<'a> {
        let index = self.next_index();
        EthercatDatagram {
            cmd: EthercatCmd::Lrw,
            index,
            address: logical_address,
            data: process_data,
            wkc: 0,
        }
    }

    /// Computes total expected Working Counter (WKC) for the cyclic LRW exchange.
    pub fn expected_cyclic_wkc(&self) -> u16 {
        self.slaves[..self.slave_count]
            .iter()
            .map(|s| s.expected_wkc)
            .sum()
    }

    /// Verifies the received WKC against expected WKC. Returns true if communication was flawless.
    pub fn verify_wkc(&self, received_wkc: u16, expected_wkc: u16) -> bool {
        received_wkc == expected_wkc
    }
}

impl<const MAX_SLAVES: usize> Default for EthercatMaster<MAX_SLAVES> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_enumeration_and_lrw() {
        let mut master = EthercatMaster::<8>::new();
        let idx0 = master.register_slave(0x0000, 0x1001).unwrap();
        let idx1 = master.register_slave(0xFFFF, 0x1002).unwrap();
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(master.slave_count, 2);

        let mut buf = [0u8; 2];
        let dgram = master
            .build_state_transition_datagram(0, EsmState::PreOp, &mut buf)
            .unwrap();
        assert_eq!(dgram.cmd, EthercatCmd::Fpwr);
        assert_eq!(dgram.address, 0x1001_0120);
        assert_eq!(dgram.data, &[0x02, 0x00]);

        // Cyclic LRW
        let mut pdo = [0xAA, 0xBB, 0xCC, 0xDD];
        let lrw_dgram = master.build_cyclic_lrw_datagram(0x0001_0000, &mut pdo);
        assert_eq!(lrw_dgram.cmd, EthercatCmd::Lrw);
        assert_eq!(lrw_dgram.address, 0x0001_0000);
        assert_eq!(master.expected_cyclic_wkc(), 6); // 2 slaves * 3
        assert!(master.verify_wkc(6, master.expected_cyclic_wkc()));
        assert!(!master.verify_wkc(3, master.expected_cyclic_wkc()));
    }
}
