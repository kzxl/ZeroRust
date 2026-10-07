//! Standard cross-language payloads for ZeroUniverse Host <-> Edge communication.

/// Commanded motion setpoint payload sent from ZeroPlatform (.NET) to ZeroRust.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionCommandPayload {
    /// Commanded timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Command Type: 1 = Position, 2 = Velocity, 3 = Homing, 4 = QuickStop, 5 = Reset.
    pub command_type: u32,
    /// Target joint/axis positions in encoder pulses / micro-units.
    pub target_positions: [i32; 6],
    /// Target joint/axis velocities in pulses/sec.
    pub target_velocities: [i32; 6],
    /// CiA 402 Controlword.
    pub controlword: u16,
    /// Reserved alignment padding.
    pub reserved: u16,
}

/// Actual machine feedback telemetry payload sent from ZeroRust to ZeroPlatform (.NET).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TelemetryPayload {
    /// Timestamp of sensor sampling in nanoseconds.
    pub timestamp_ns: u64,
    /// Actual measured feedback positions for up to 6 axes.
    pub actual_positions: [i32; 6],
    /// Actual measured feedback velocities for up to 6 axes.
    pub actual_velocities: [i32; 6],
    /// CiA 402 Statuswords for up to 6 drives.
    pub statuswords: [u16; 6],
    /// CiA 402 Drive state codes for up to 6 drives.
    pub drive_states: [u8; 6],
    /// Error / Fault code (0 = OK).
    pub fault_code: u16,
    /// Reserved alignment padding to fill 80 bytes.
    pub reserved: [u8; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_alignment() {
        assert_eq!(core::mem::size_of::<MotionCommandPayload>(), 64);
        assert_eq!(core::mem::size_of::<TelemetryPayload>(), 80);
    }
}
