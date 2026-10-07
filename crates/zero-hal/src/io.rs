//! Industrial digital input and output abstractions with debouncing.

use zero_core::error::ZeroResult;

/// Trait for digital input pins or virtual inputs.
pub trait DigitalInput {
    /// Returns true if the digital input is logical HIGH.
    fn is_high(&self) -> bool;

    /// Returns true if the digital input is logical LOW.
    #[inline]
    fn is_low(&self) -> bool {
        !self.is_high()
    }
}

/// Trait for digital output pins or virtual relays.
pub trait DigitalOutput {
    /// Sets output to logical HIGH.
    fn set_high(&mut self) -> ZeroResult<()>;

    /// Sets output to logical LOW.
    fn set_low(&mut self) -> ZeroResult<()>;

    /// Sets output to the specified boolean state.
    #[inline]
    fn set_state(&mut self, state: bool) -> ZeroResult<()> {
        if state {
            self.set_high()
        } else {
            self.set_low()
        }
    }
}

/// In-memory mock pin for testing and simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MockPin {
    state: bool,
}

impl MockPin {
    /// Creates a new mock pin initialized to LOW.
    pub const fn new() -> Self {
        Self { state: false }
    }

    /// Creates a mock pin with an initial state.
    pub const fn with_state(state: bool) -> Self {
        Self { state }
    }
}

impl DigitalInput for MockPin {
    #[inline]
    fn is_high(&self) -> bool {
        self.state
    }
}

impl DigitalOutput for MockPin {
    #[inline]
    fn set_high(&mut self) -> ZeroResult<()> {
        self.state = true;
        Ok(())
    }

    #[inline]
    fn set_low(&mut self) -> ZeroResult<()> {
        self.state = false;
        Ok(())
    }
}

/// Digital signal debouncer to filter out contact bounce and electrical noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Debouncer {
    stable_state: bool,
    integrator: u8,
    threshold: u8,
}

impl Debouncer {
    /// Creates a new debouncer with initial state and stability sample threshold.
    pub const fn new(initial_state: bool, threshold: u8) -> Self {
        Self {
            stable_state: initial_state,
            integrator: if initial_state { threshold } else { 0 },
            threshold,
        }
    }

    /// Feeds a new raw sample from the pin and returns the current filtered stable state.
    pub fn update(&mut self, raw_sample: bool) -> bool {
        if raw_sample {
            if self.integrator < self.threshold {
                self.integrator += 1;
            }
        } else if self.integrator > 0 {
            self.integrator -= 1;
        }

        if self.integrator >= self.threshold {
            self.stable_state = true;
        } else if self.integrator == 0 {
            self.stable_state = false;
        }

        self.stable_state
    }

    /// Returns the current debounced stable state.
    #[inline]
    pub const fn state(&self) -> bool {
        self.stable_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debouncer_filtering() {
        let mut debouncer = Debouncer::new(false, 3);
        assert!(!debouncer.state());

        // Glitch 1: brief 1-cycle spike
        assert!(!debouncer.update(true));
        assert!(!debouncer.update(false));
        assert!(!debouncer.state());

        // Genuine transition: 3 consecutive HIGHs
        assert!(!debouncer.update(true));
        assert!(!debouncer.update(true));
        assert!(debouncer.update(true)); // Threshold reached!
        assert!(debouncer.state());
    }

    #[test]
    fn test_mock_pin() {
        let mut pin = MockPin::new();
        assert!(pin.is_low());
        pin.set_high().unwrap();
        assert!(pin.is_high());
        pin.set_state(false).unwrap();
        assert!(pin.is_low());
    }
}
