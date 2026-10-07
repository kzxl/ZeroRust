//! Linux Systemd service control and status inspection.

/// Actions performable on a Linux system service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceAction {
    /// Start the service immediately.
    Start,
    /// Stop the running service.
    Stop,
    /// Restart the service.
    Restart,
    /// Reload configuration without terminating workers.
    Reload,
    /// Enable service to start automatically at boot.
    Enable,
    /// Disable service from starting at boot.
    Disable,
}

impl ServiceAction {
    /// Returns the systemctl verb argument corresponding to this action.
    pub const fn as_verb(&self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Stop => "stop",
            Self::Restart => "restart",
            Self::Reload => "reload",
            Self::Enable => "enable",
            Self::Disable => "disable",
        }
    }
}

/// Operational state of a systemd unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceState {
    /// Service is up and running (`active (running)`).
    Active,
    /// Service is stopped / dead (`inactive (dead)`).
    Inactive,
    /// Service exited with an error (`failed`).
    Failed,
    /// Service is currently reloading configuration.
    Reloading,
    /// State could not be determined.
    #[default]
    Unknown,
}

impl ServiceState {
    /// Parses standard `systemctl is-active <unit>` one-line output.
    pub fn from_is_active(output: &str) -> Self {
        match output.trim() {
            "active" => Self::Active,
            "inactive" | "dead" => Self::Inactive,
            "failed" => Self::Failed,
            "reloading" => Self::Reloading,
            _ => Self::Unknown,
        }
    }
}

/// Helper to build and inspect systemctl invocations.
pub struct SystemdManager;

impl SystemdManager {
    /// Builds the standard argument array for invoking `systemctl <action> <service>.service`.
    pub const fn build_command_args(action: ServiceAction, service_name: &str) -> [&str; 2] {
        [action.as_verb(), service_name]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systemd_status_parsing() {
        assert_eq!(
            ServiceState::from_is_active("active\n"),
            ServiceState::Active
        );
        assert_eq!(
            ServiceState::from_is_active("inactive"),
            ServiceState::Inactive
        );
        assert_eq!(ServiceState::from_is_active("failed"), ServiceState::Failed);
        assert_eq!(
            ServiceState::from_is_active("something_else"),
            ServiceState::Unknown
        );
    }

    #[test]
    fn test_systemd_args() {
        let args = SystemdManager::build_command_args(ServiceAction::Restart, "caddy");
        assert_eq!(args[0], "restart");
        assert_eq!(args[1], "caddy");
    }
}
