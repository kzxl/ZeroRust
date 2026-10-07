//! CiA 402 Servo Drive and Motion Control Device Profile.

use zero_core::error::{ZeroError, ZeroResult};

/// CiA 402 Drive Finite State Machine (FSM) states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriveState {
    /// Controller is starting up or in hardware reset.
    #[default]
    NotReadyToSwitchOn,
    /// High voltage may be applied, drive parameters can be configured.
    SwitchOnDisabled,
    /// Drive is ready to switch on power electronics.
    ReadyToSwitchOn,
    /// Power section is energized, motor is unexcited.
    SwitchedOn,
    /// Drive is enabled, motor torque/power active, motion can be commanded.
    OperationEnabled,
    /// Quick stop command executing.
    QuickStopActive,
    /// Drive detected a fault and is executing fault reaction.
    FaultReactionActive,
    /// Fault state; motor disabled until Fault Reset command.
    Fault,
}

/// Standard CiA 402 Operation Modes (Object 0x6060 / 0x6061).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum OperationMode {
    /// No mode assigned (0).
    NoMode = 0,
    /// Profile Position Mode (PPM = 1).
    ProfilePosition = 1,
    /// Profile Velocity Mode (PVM = 3).
    ProfileVelocity = 3,
    /// Profile Torque Mode (PTM = 4).
    ProfileTorque = 4,
    /// Homing Mode (HM = 6).
    Homing = 6,
    /// Cyclic Synchronous Position (CSP = 8).
    CyclicSynchronousPosition = 8,
    /// Cyclic Synchronous Velocity (CSV = 9).
    CyclicSynchronousVelocity = 9,
    /// Cyclic Synchronous Torque (CST = 10).
    CyclicSynchronousTorque = 10,
}

/// Helper for constructing and inspecting CiA 402 Controlwords (Object 0x6040).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ControlWord(pub u16);

impl ControlWord {
    /// Command: Shutdown (0x0006).
    pub const SHUTDOWN: Self = Self(0x0006);
    /// Command: Switch On (0x0007).
    pub const SWITCH_ON: Self = Self(0x0007);
    /// Command: Enable Operation (0x000F).
    pub const ENABLE_OPERATION: Self = Self(0x000F);
    /// Command: Disable Voltage (0x0000).
    pub const DISABLE_VOLTAGE: Self = Self(0x0000);
    /// Command: Quick Stop (0x0002).
    pub const QUICK_STOP: Self = Self(0x0002);
    /// Command: Fault Reset (0x0080).
    pub const FAULT_RESET: Self = Self(0x0080);

    /// Raw 16-bit controlword value.
    #[inline]
    pub const fn raw(&self) -> u16 {
        self.0
    }
}

/// Helper for decoding CiA 402 Statuswords (Object 0x6041).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatusWord(pub u16);

impl StatusWord {
    /// Creates a new `StatusWord` from raw 16-bit register value.
    #[inline]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Bit 0: Ready to switch on.
    #[inline]
    pub const fn ready_to_switch_on(&self) -> bool {
        (self.0 & (1 << 0)) != 0
    }

    /// Bit 1: Switched on.
    #[inline]
    pub const fn switched_on(&self) -> bool {
        (self.0 & (1 << 1)) != 0
    }

    /// Bit 2: Operation enabled.
    #[inline]
    pub const fn operation_enabled(&self) -> bool {
        (self.0 & (1 << 2)) != 0
    }

    /// Bit 3: Fault condition.
    #[inline]
    pub const fn fault(&self) -> bool {
        (self.0 & (1 << 3)) != 0
    }

    /// Bit 4: Voltage enabled.
    #[inline]
    pub const fn voltage_enabled(&self) -> bool {
        (self.0 & (1 << 4)) != 0
    }

    /// Bit 5: Quick stop inactive (low active).
    #[inline]
    pub const fn quick_stop_inactive(&self) -> bool {
        (self.0 & (1 << 5)) != 0
    }

    /// Bit 6: Switch on disabled.
    #[inline]
    pub const fn switch_on_disabled(&self) -> bool {
        (self.0 & (1 << 6)) != 0
    }

    /// Bit 7: Warning.
    #[inline]
    pub const fn warning(&self) -> bool {
        (self.0 & (1 << 7)) != 0
    }

    /// Bit 10: Target reached.
    #[inline]
    pub const fn target_reached(&self) -> bool {
        (self.0 & (1 << 10)) != 0
    }

    /// Decodes the current `DriveState` from the bitmask according to CiA 402 specification.
    pub fn decode_state(&self) -> DriveState {
        let mask_fault = self.0 & 0x004F;
        if mask_fault == 0x0008 || mask_fault == 0x0028 {
            return DriveState::Fault;
        }

        let mask_quick_stop = self.0 & 0x006F;
        if mask_quick_stop == 0x0007 {
            return DriveState::QuickStopActive;
        }

        let mask = self.0 & 0x006F;
        match mask {
            0x0000 => DriveState::NotReadyToSwitchOn,
            0x0040 => DriveState::SwitchOnDisabled,
            0x0021 => DriveState::ReadyToSwitchOn,
            0x0023 => DriveState::SwitchedOn,
            0x0027 => DriveState::OperationEnabled,
            0x000F => DriveState::FaultReactionActive,
            _ => DriveState::NotReadyToSwitchOn,
        }
    }
}

/// CiA 402 Servo Drive Controller.
pub struct Cia402Drive {
    node_id: u8,
    state: DriveState,
    target_mode: OperationMode,
    actual_mode: OperationMode,
    target_position: i32,
    actual_position: i32,
}

impl Cia402Drive {
    /// Creates a new drive controller for the given node ID.
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            state: DriveState::SwitchOnDisabled,
            target_mode: OperationMode::CyclicSynchronousPosition,
            actual_mode: OperationMode::NoMode,
            target_position: 0,
            actual_position: 0,
        }
    }

    /// Returns the assigned CANopen Node ID.
    #[inline]
    pub const fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Returns the current known drive state.
    #[inline]
    pub const fn current_state(&self) -> DriveState {
        self.state
    }

    /// Updates internal state from a received StatusWord (0x6041).
    pub fn update_statusword(&mut self, statusword: StatusWord) {
        self.state = statusword.decode_state();
    }

    /// Computes the required ControlWord transition to move towards `target_state`.
    pub fn command_transition_to(&self, target_state: DriveState) -> ZeroResult<ControlWord> {
        match (self.state, target_state) {
            (DriveState::Fault, _) => Ok(ControlWord::FAULT_RESET),
            (DriveState::SwitchOnDisabled, DriveState::ReadyToSwitchOn)
            | (DriveState::SwitchOnDisabled, DriveState::SwitchedOn)
            | (DriveState::SwitchOnDisabled, DriveState::OperationEnabled) => {
                Ok(ControlWord::SHUTDOWN)
            }

            (DriveState::ReadyToSwitchOn, DriveState::SwitchedOn)
            | (DriveState::ReadyToSwitchOn, DriveState::OperationEnabled) => {
                Ok(ControlWord::SWITCH_ON)
            }

            (DriveState::SwitchedOn, DriveState::OperationEnabled) => {
                Ok(ControlWord::ENABLE_OPERATION)
            }

            (DriveState::OperationEnabled, DriveState::SwitchedOn) => Ok(ControlWord::SWITCH_ON),
            (DriveState::OperationEnabled, DriveState::ReadyToSwitchOn) => {
                Ok(ControlWord::SHUTDOWN)
            }
            (DriveState::OperationEnabled, DriveState::QuickStopActive) => {
                Ok(ControlWord::QUICK_STOP)
            }

            (s, t) if s == t => Ok(ControlWord::ENABLE_OPERATION),
            _ => Err(ZeroError::InvalidArgument),
        }
    }

    /// Sets the target operation mode.
    pub fn set_target_mode(&mut self, mode: OperationMode) {
        self.target_mode = mode;
    }

    /// Sets the actual reported operation mode (Object 0x6061).
    pub fn set_actual_mode(&mut self, mode: OperationMode) {
        self.actual_mode = mode;
    }

    /// Returns the actual reported operation mode.
    pub const fn actual_mode(&self) -> OperationMode {
        self.actual_mode
    }

    /// Sets the target setpoint position in encoder counts / pulses.
    pub fn set_target_position(&mut self, pos: i32) {
        self.target_position = pos;
    }

    /// Returns the last commanded target position.
    pub const fn target_position(&self) -> i32 {
        self.target_position
    }

    /// Updates the actual feedback position from the motor encoder.
    pub fn update_actual_position(&mut self, pos: i32) {
        self.actual_position = pos;
    }

    /// Returns the actual feedback position.
    pub const fn actual_position(&self) -> i32 {
        self.actual_position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statusword_decoding() {
        // Status 0x0240 -> Switch on disabled (bit 6 = 1)
        let sw_disabled = StatusWord::new(0x0240);
        assert_eq!(sw_disabled.decode_state(), DriveState::SwitchOnDisabled);

        // Status 0x0221 -> Ready to switch on (bit 0 = 1, bit 5 = 1)
        let sw_ready = StatusWord::new(0x0221);
        assert_eq!(sw_ready.decode_state(), DriveState::ReadyToSwitchOn);

        // Status 0x0227 -> Operation enabled (bit 0,1,2 = 1, bit 5 = 1)
        let sw_op = StatusWord::new(0x0227);
        assert_eq!(sw_op.decode_state(), DriveState::OperationEnabled);
    }

    #[test]
    fn test_transition_commands() {
        let mut drive = Cia402Drive::new(1);
        drive.update_statusword(StatusWord::new(0x0240)); // SwitchOnDisabled

        let cmd1 = drive
            .command_transition_to(DriveState::OperationEnabled)
            .unwrap();
        assert_eq!(cmd1, ControlWord::SHUTDOWN);

        drive.update_statusword(StatusWord::new(0x0221)); // ReadyToSwitchOn
        let cmd2 = drive
            .command_transition_to(DriveState::OperationEnabled)
            .unwrap();
        assert_eq!(cmd2, ControlWord::SWITCH_ON);

        drive.update_statusword(StatusWord::new(0x0223)); // SwitchedOn
        let cmd3 = drive
            .command_transition_to(DriveState::OperationEnabled)
            .unwrap();
        assert_eq!(cmd3, ControlWord::ENABLE_OPERATION);
    }
}
