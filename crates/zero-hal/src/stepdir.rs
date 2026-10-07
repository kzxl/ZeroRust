//! Step and Direction pulse train generator for stepper and servo drives.

/// Step/Dir pulse generator using a high-precision fractional accumulator (DDA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepDirGenerator {
    sample_rate_hz: u32,
    target_frequency_hz: u32,
    direction_forward: bool,
    accumulator: u32,
    step_active: bool,
    total_steps: i64,
}

impl StepDirGenerator {
    /// Creates a new generator with the specified cyclic sample rate (e.g. 100_000 Hz or 10_000 Hz).
    pub const fn new(sample_rate_hz: u32) -> Self {
        assert!(sample_rate_hz > 0, "Sample rate must be positive");
        Self {
            sample_rate_hz,
            target_frequency_hz: 0,
            direction_forward: true,
            accumulator: 0,
            step_active: false,
            total_steps: 0,
        }
    }

    /// Sets the target step frequency and motion direction.
    pub fn set_velocity(&mut self, frequency_hz: u32, forward: bool) {
        // Clamp frequency to Nyquist limit (sample_rate / 2)
        self.target_frequency_hz = frequency_hz.min(self.sample_rate_hz / 2);
        self.direction_forward = forward;
    }

    /// Advances the generator by one tick of the sample clock.
    ///
    /// Returns `(step_pin_state, dir_pin_state)`.
    pub fn tick(&mut self) -> (bool, bool) {
        if self.target_frequency_hz == 0 {
            return (false, self.direction_forward);
        }

        self.accumulator = self.accumulator.wrapping_add(self.target_frequency_hz);
        if self.accumulator >= self.sample_rate_hz {
            self.accumulator -= self.sample_rate_hz;

            if self.direction_forward {
                self.total_steps += 1;
            } else {
                self.total_steps -= 1;
            }

            (true, self.direction_forward)
        } else {
            (false, self.direction_forward)
        }
    }

    /// Returns the accumulated step position.
    #[inline]
    pub const fn total_steps(&self) -> i64 {
        self.total_steps
    }

    /// Returns the current direction state.
    #[inline]
    pub const fn direction(&self) -> bool {
        self.direction_forward
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_dir_generation() {
        let mut gen = StepDirGenerator::new(10_000); // 10 kHz base
        gen.set_velocity(1_000, true); // 1 kHz step target

        let mut step_count = 0;
        for _ in 0..10_000 {
            let (step, dir) = gen.tick();
            assert!(dir);
            if step {
                step_count += 1;
            }
        }

        // Over 1 second at 1 kHz, should have approximately 1000 steps
        let diff: i32 = step_count - 1000;
        assert!(diff.abs() <= 2);
        assert_eq!(gen.total_steps(), step_count as i64);
    }
}
