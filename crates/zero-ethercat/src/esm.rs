//! EtherCAT State Machine (ESM) definitions, AL Control / Status register mapping, and transitions.

use zero_core::error::{ZeroError, ZeroResult};

/// EtherCAT State Machine (ESM) States.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EsmState {
    /// Initialization State (0x01). No communication with ESC application.
    Init = 0x01,
    /// Pre-Operational State (0x02). Mailbox communication active; no process data.
    PreOp = 0x02,
    /// Bootstrap State (0x03). Firmware update / FoE active.
    Boot = 0x03,
    /// Safe-Operational State (0x04). Mailbox and inputs active; outputs in safe state.
    SafeOp = 0x04,
    /// Operational State (0x08). Full cyclic process data and mailbox exchange.
    Op = 0x08,
}

impl TryFrom<u8> for EsmState {
    type Error = ZeroError;

    fn try_from(val: u8) -> Result<Self, Self::Error> {
        match val & 0x0F {
            0x01 => Ok(Self::Init),
            0x02 => Ok(Self::PreOp),
            0x03 => Ok(Self::Boot),
            0x04 => Ok(Self::SafeOp),
            0x08 => Ok(Self::Op),
            _ => Err(ZeroError::InvalidArgument),
        }
    }
}

/// ESC Register Addresses for AL Control and Status.
pub struct EscRegister;

impl EscRegister {
    /// Configured Station Address (0x0010, 2 bytes).
    pub const STATION_ADDRESS: u16 = 0x0010;
    /// DL Status Register (0x0110, 2 bytes).
    pub const DL_STATUS: u16 = 0x0110;
    /// AL Control Register (0x0120, 2 bytes).
    pub const AL_CONTROL: u16 = 0x0120;
    /// AL Status Register (0x0130, 2 bytes).
    pub const AL_STATUS: u16 = 0x0130;
    /// AL Status Code Register (0x0134, 2 bytes).
    pub const AL_STATUS_CODE: u16 = 0x0134;
    /// PDI Control Register (0x0140, 2 bytes).
    pub const PDI_CONTROL: u16 = 0x0140;
    /// Sync Manager 0 Configuration (Mailbox Out, 0x0800).
    pub const SM0_CONFIG: u16 = 0x0800;
    /// Sync Manager 1 Configuration (Mailbox In, 0x0808).
    pub const SM1_CONFIG: u16 = 0x0808;
    /// Sync Manager 2 Configuration (Outputs PDO, 0x0810).
    pub const SM2_CONFIG: u16 = 0x0810;
    /// Sync Manager 3 Configuration (Inputs PDO, 0x0818).
    pub const SM3_CONFIG: u16 = 0x0818;
    /// Distributed Clocks System Time (0x0910, 8 bytes).
    pub const DC_SYSTEM_TIME: u16 = 0x0910;
    /// Distributed Clocks System Time Offset (0x0920, 8 bytes).
    pub const DC_SYSTEM_TIME_OFFSET: u16 = 0x0920;
    /// Distributed Clocks System Time Delay (0x0928, 4 bytes).
    pub const DC_SYSTEM_TIME_DELAY: u16 = 0x0928;
    /// SYNC0 Cycle Time (0x09A0, 4 bytes).
    pub const DC_SYNC0_CYCLE_TIME: u16 = 0x09A0;
}

/// AL Status parsing result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlStatus {
    /// Current state of the slave.
    pub state: EsmState,
    /// True if an error flag is raised by the slave (bit 4).
    pub error_flag: bool,
}

impl AlStatus {
    /// Parses AL Status raw 16-bit word from register 0x0130.
    pub fn parse(raw: u16) -> ZeroResult<Self> {
        let state = EsmState::try_from((raw & 0x0F) as u8)?;
        let error_flag = (raw & 0x10) != 0;
        Ok(Self { state, error_flag })
    }

    /// Formats an AL Control word to request a state and optional error acknowledge.
    pub fn build_control_word(target_state: EsmState, error_ack: bool) -> u16 {
        let mut val = target_state as u16;
        if error_ack {
            val |= 0x0010;
        }
        val
    }
}

/// Standard AL Status Codes (EtherCAT error codes from register 0x0134).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum AlStatusCode {
    /// No error (0x0000).
    NoError = 0x0000,
    /// Unspecified error (0x0001).
    UnspecifiedError = 0x0001,
    /// Invalid requested state change (0x0011).
    InvalidRequestedStateChange = 0x0011,
    /// Unknown requested state (0x0012).
    UnknownRequestedState = 0x0012,
    /// Bootstrap not supported (0x0013).
    BootstrapNotSupported = 0x0013,
    /// Invalid mailbox configuration in Pre-Operational (0x0016).
    InvalidMailboxConfigurationPreOp = 0x0016,
    /// Invalid Sync Manager configuration (0x001D).
    InvalidSyncManagerConfiguration = 0x001D,
    /// Invalid output configuration (0x001E).
    InvalidOutputConfiguration = 0x001E,
    /// Invalid input configuration (0x001F).
    InvalidInputConfiguration = 0x001F,
    /// Watchdog expired in Safe-Operational or Operational (0x001B).
    WatchdogProcessData = 0x001B,
    /// DC Sync0 / Sync1 cycle time invalid (0x002C).
    InvalidDcSyncCycle = 0x002C,
    /// Unknown vendor or reserved error code.
    Other(u16),
}

impl From<u16> for AlStatusCode {
    fn from(val: u16) -> Self {
        match val {
            0x0000 => Self::NoError,
            0x0001 => Self::UnspecifiedError,
            0x0011 => Self::InvalidRequestedStateChange,
            0x0012 => Self::UnknownRequestedState,
            0x0013 => Self::BootstrapNotSupported,
            0x0016 => Self::InvalidMailboxConfigurationPreOp,
            0x001D => Self::InvalidSyncManagerConfiguration,
            0x001E => Self::InvalidOutputConfiguration,
            0x001F => Self::InvalidInputConfiguration,
            0x001B => Self::WatchdogProcessData,
            0x002C => Self::InvalidDcSyncCycle,
            other => Self::Other(other),
        }
    }
}

/// Checks if a transition from `from` state to `to` state is valid according to ETG.1000.
pub fn is_valid_esm_transition(from: EsmState, to: EsmState) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        // Forward transitions
        (EsmState::Init, EsmState::PreOp) => true,
        (EsmState::Init, EsmState::Boot) => true,
        (EsmState::PreOp, EsmState::SafeOp) => true,
        (EsmState::SafeOp, EsmState::Op) => true,
        // Backward transitions (always allowed directly or down to Init)
        (EsmState::Op, EsmState::SafeOp) => true,
        (EsmState::Op, EsmState::PreOp) => true,
        (EsmState::Op, EsmState::Init) => true,
        (EsmState::SafeOp, EsmState::PreOp) => true,
        (EsmState::SafeOp, EsmState::Init) => true,
        (EsmState::PreOp, EsmState::Init) => true,
        (EsmState::Boot, EsmState::Init) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esm_transitions() {
        assert!(is_valid_esm_transition(EsmState::Init, EsmState::PreOp));
        assert!(is_valid_esm_transition(EsmState::PreOp, EsmState::SafeOp));
        assert!(is_valid_esm_transition(EsmState::SafeOp, EsmState::Op));
        // Direct jump from Init to Op is invalid without going through PreOp & SafeOp
        assert!(!is_valid_esm_transition(EsmState::Init, EsmState::Op));
        // Drop down is valid
        assert!(is_valid_esm_transition(EsmState::Op, EsmState::SafeOp));
        assert!(is_valid_esm_transition(EsmState::Op, EsmState::Init));
    }

    #[test]
    fn test_al_status_and_control() {
        let control = AlStatus::build_control_word(EsmState::PreOp, false);
        assert_eq!(control, 0x0002);

        let control_ack = AlStatus::build_control_word(EsmState::Init, true);
        assert_eq!(control_ack, 0x0011);

        let parsed = AlStatus::parse(0x0014).unwrap(); // SafeOp + Error
        assert_eq!(parsed.state, EsmState::SafeOp);
        assert!(parsed.error_flag);

        let parsed_clean = AlStatus::parse(0x0008).unwrap(); // Op clean
        assert_eq!(parsed_clean.state, EsmState::Op);
        assert!(!parsed_clean.error_flag);
    }
}
