//! Distributed Clocks (DC) engine for sub-microsecond hardware synchronization.

use zero_core::error::{ZeroError, ZeroResult};

/// Distributed Clocks configuration for an individual slave device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcConfig {
    /// SYNC0 cycle time in nanoseconds (e.g. 1_000_000 ns = 1 ms).
    pub sync0_cycle_ns: u32,
    /// SYNC1 cycle time in nanoseconds (0 if unused).
    pub sync1_cycle_ns: u32,
    /// SYNC0 shift time from start of cycle in nanoseconds.
    pub sync0_shift_ns: i32,
    /// Enable SYNC0 output generation.
    pub enable_sync0: bool,
    /// Enable SYNC1 output generation.
    pub enable_sync1: bool,
}

impl Default for DcConfig {
    fn default() -> Self {
        Self {
            sync0_cycle_ns: 1_000_000, // 1 ms default cycle
            sync1_cycle_ns: 0,
            sync0_shift_ns: 0,
            enable_sync0: true,
            enable_sync1: false,
        }
    }
}

/// Port timestamp snapshot captured during DC propagation delay measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PortTimestamps {
    /// Port 0 receive timestamp (nanoseconds).
    pub port0_receive_ns: u32,
    /// Port 1 receive timestamp (nanoseconds).
    pub port1_receive_ns: u32,
    /// Port 2 receive timestamp (nanoseconds).
    pub port2_receive_ns: u32,
    /// Port 3 receive timestamp (nanoseconds).
    pub port3_receive_ns: u32,
}

/// Distributed Clocks synchronization engine.
#[derive(Debug, Clone)]
pub struct DcEngine {
    /// Reference clock system time (nanoseconds).
    pub reference_time_ns: u64,
    /// Master cycle time in nanoseconds.
    pub cycle_time_ns: u32,
}

impl DcEngine {
    /// Creates a new DC Engine with specified master cycle time in nanoseconds.
    pub fn new(cycle_time_ns: u32) -> Self {
        Self {
            reference_time_ns: 0,
            cycle_time_ns,
        }
    }

    /// Calculates the line propagation delay between node A and node B.
    ///
    /// For a full-duplex Ethernet segment between port A1 and port B0:
    /// `forward_time = time_b_in - time_a_out`
    /// `return_time = time_a_in - time_b_out`
    /// Delay = (forward_time + return_time) / 2
    pub fn calculate_segment_delay(
        time_a_out: u32,
        time_b_in: u32,
        time_b_out: u32,
        time_a_in: u32,
    ) -> u32 {
        let forward_delta = time_b_in.wrapping_sub(time_a_out);
        let return_delta = time_a_in.wrapping_sub(time_b_out);
        (forward_delta + return_delta) / 2
    }

    /// Computes the System Time Offset register (0x0920) value for a slave.
    ///
    /// `system_time_offset = reference_system_time - (slave_local_time - propagation_delay)`
    pub fn compute_system_time_offset(
        reference_time_ns: u64,
        slave_local_time_ns: u64,
        propagation_delay_ns: u32,
    ) -> u64 {
        let adjusted_slave = slave_local_time_ns.saturating_sub(propagation_delay_ns as u64);
        reference_time_ns.wrapping_sub(adjusted_slave)
    }

    /// Calculates the next synchronized start time for SYNC0 activation.
    ///
    /// Aligns `current_time_ns` to the nearest future multiple of `cycle_time_ns`,
    /// plus a safety margin (e.g. 2 cycles) and the optional shift offset.
    pub fn calculate_start_time(
        current_time_ns: u64,
        cycle_time_ns: u32,
        shift_ns: i32,
        safety_cycles: u32,
    ) -> ZeroResult<u64> {
        if cycle_time_ns == 0 {
            return Err(ZeroError::InvalidArgument);
        }
        let cycle = cycle_time_ns as u64;
        let remainder = current_time_ns % cycle;
        let base_aligned = current_time_ns - remainder + (safety_cycles as u64 * cycle);

        let final_time = if shift_ns >= 0 {
            base_aligned + shift_ns as u64
        } else {
            base_aligned.saturating_sub((-shift_ns) as u64)
        };

        Ok(final_time)
    }

    /// Builds the 4-byte Activation Register (0x0990) word from config.
    pub fn activation_register(config: &DcConfig) -> u16 {
        let mut reg: u16 = 0;
        if config.enable_sync0 {
            reg |= 0x0001; // Bit 0: SYNC0 Generation
        }
        if config.enable_sync1 {
            reg |= 0x0002; // Bit 1: SYNC1 Generation
        }
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_delay_calculation() {
        // A sends at 1000, B receives at 1050 (50ns wire delay)
        // B sends at 2000, A receives at 2050 (50ns return wire delay)
        let delay = DcEngine::calculate_segment_delay(1000, 1050, 2000, 2050);
        assert_eq!(delay, 50);
    }

    #[test]
    fn test_sync_start_time_alignment() {
        let current_time: u64 = 1_250_000; // 1.25 ms
        let cycle_ns: u32 = 1_000_000; // 1.00 ms
        let shift_ns: i32 = 100_000; // 100 us shift

        // With 2 safety cycles:
        // base = 1_250_000 - 250_000 + 2_000_000 = 3_000_000 ns
        // final = 3_000_000 + 100_000 = 3_100_000 ns
        let start = DcEngine::calculate_start_time(current_time, cycle_ns, shift_ns, 2).unwrap();
        assert_eq!(start, 3_100_000);
        assert_eq!((start - 100_000) % (cycle_ns as u64), 0);
    }
}
