//! Synthetic input injection interface and backends.

use crate::events::{KeyboardEvent, MouseEvent};

/// Errors encountered during synthetic input injection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    /// Operating system rejected input injection.
    InjectionFailed(String),
    /// Permission denied (e.g. UAC secure desktop or UIPI restriction).
    PermissionDenied,
    /// Invalid coordinate or parameter.
    InvalidParameter,
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InjectionFailed(msg) => write!(f, "Input injection failed: {}", msg),
            Self::PermissionDenied => write!(f, "Input injection permission denied"),
            Self::InvalidParameter => write!(f, "Invalid input parameter"),
        }
    }
}

impl std::error::Error for InputError {}

/// Abstract synthetic input injector interface.
pub trait InputInjector: Send {
    /// Injects a synthetic mouse movement, button, or scroll event.
    fn inject_mouse(&mut self, event: &MouseEvent) -> Result<(), InputError>;

    /// Injects a synthetic keyboard key press or release event.
    fn inject_keyboard(&mut self, event: &KeyboardEvent) -> Result<(), InputError>;
}

/// Headless mock injector that records all dispatched events for verification.
#[derive(Debug, Default)]
pub struct MockInjector {
    /// History of dispatched mouse events.
    pub mouse_events: Vec<MouseEvent>,
    /// History of dispatched keyboard events.
    pub keyboard_events: Vec<KeyboardEvent>,
}

impl MockInjector {
    /// Creates a new mock injector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears recorded event history.
    pub fn clear(&mut self) {
        self.mouse_events.clear();
        self.keyboard_events.clear();
    }
}

impl InputInjector for MockInjector {
    fn inject_mouse(&mut self, event: &MouseEvent) -> Result<(), InputError> {
        self.mouse_events.push(*event);
        Ok(())
    }

    fn inject_keyboard(&mut self, event: &KeyboardEvent) -> Result<(), InputError> {
        self.keyboard_events.push(*event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ElementState, MouseButton};

    #[test]
    fn test_mock_injector() {
        let mut injector = MockInjector::new();
        let m = MouseEvent::Button {
            button: MouseButton::Left,
            state: ElementState::Pressed,
        };
        injector.inject_mouse(&m).expect("Inject mouse");

        let k = KeyboardEvent {
            scancode: 0x04,
            state: ElementState::Pressed,
            modifiers: 0,
        };
        injector.inject_keyboard(&k).expect("Inject key");

        assert_eq!(injector.mouse_events.len(), 1);
        assert_eq!(injector.keyboard_events.len(), 1);
    }
}
