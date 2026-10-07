//! Anti-stuck modifier and key safety guard state machine.

use crate::events::{ElementState, KeyboardEvent};

/// Tracks active held keys and guarantees zero stuck keys upon session blur or disconnect.
#[derive(Debug, Clone)]
pub struct SafetyGuard {
    /// 256-bit bitmask tracking which USB HID scancodes are currently held down.
    active_keys: [u64; 4],
    /// Monotonic timestamp in milliseconds of the last received input or heartbeat.
    last_activity_ms: u64,
    /// Heartbeat timeout threshold in milliseconds (default: 2000 ms).
    timeout_threshold_ms: u64,
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self {
            active_keys: [0; 4],
            last_activity_ms: 0,
            timeout_threshold_ms: 2000,
        }
    }
}

impl SafetyGuard {
    /// Creates a new safety guard with custom timeout.
    pub fn new(timeout_threshold_ms: u64) -> Self {
        Self {
            active_keys: [0; 4],
            last_activity_ms: 0,
            timeout_threshold_ms,
        }
    }

    /// Registers a key event and updates the active key tracking mask.
    pub fn register_key(&mut self, event: &KeyboardEvent, current_time_ms: u64) {
        self.last_activity_ms = current_time_ms;
        let idx = (event.scancode / 64) as usize;
        let bit = event.scancode % 64;

        match event.state {
            ElementState::Pressed => {
                self.active_keys[idx] |= 1 << bit;
            }
            ElementState::Released => {
                self.active_keys[idx] &= !(1 << bit);
            }
        }
    }

    /// Checks if a session heartbeat timeout has expired.
    pub fn is_timed_out(&self, current_time_ms: u64) -> bool {
        if self.last_activity_ms == 0 {
            return false;
        }
        current_time_ms.saturating_sub(self.last_activity_ms) > self.timeout_threshold_ms
    }

    /// Generates synthetic `Released` events for all currently held keys to prevent stuck keys.
    pub fn release_all_keys(&mut self) -> Vec<KeyboardEvent> {
        let mut release_events = Vec::new();

        for idx in 0..4 {
            let mut mask = self.active_keys[idx];
            while mask != 0 {
                let bit = mask.trailing_zeros();
                let scancode = (idx * 64 + bit as usize) as u8;
                release_events.push(KeyboardEvent {
                    scancode,
                    state: ElementState::Released,
                    modifiers: 0,
                });
                mask &= !(1 << bit);
            }
            self.active_keys[idx] = 0;
        }

        release_events
    }

    /// Checks if any key is currently recorded as pressed.
    pub fn has_pressed_keys(&self) -> bool {
        self.active_keys.iter().any(|&mask| mask != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scancodes::hid;

    #[test]
    fn test_safety_guard_release() {
        let mut guard = SafetyGuard::new(1000);

        guard.register_key(
            &KeyboardEvent {
                scancode: hid::KEY_LEFT_CTRL,
                state: ElementState::Pressed,
                modifiers: 0x02,
            },
            100,
        );
        guard.register_key(
            &KeyboardEvent {
                scancode: hid::KEY_A,
                state: ElementState::Pressed,
                modifiers: 0x02,
            },
            110,
        );

        assert!(guard.has_pressed_keys());

        // Normal release of 'A'
        guard.register_key(
            &KeyboardEvent {
                scancode: hid::KEY_A,
                state: ElementState::Released,
                modifiers: 0x02,
            },
            120,
        );
        assert!(guard.has_pressed_keys()); // Ctrl is still held!

        // Emergency session disconnect trigger
        let releases = guard.release_all_keys();
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].scancode, hid::KEY_LEFT_CTRL);
        assert_eq!(releases[0].state, ElementState::Released);
        assert!(!guard.has_pressed_keys());
    }
}
