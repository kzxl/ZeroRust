//! IEC 61131-3 Standard Function Blocks (TON, TOF, TP, CTU, CTD, and PID_Compact).

/// TON - On-Delay Timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ton {
    /// Preset Time in milliseconds.
    pub pt_ms: u32,
    /// Elapsed Time in milliseconds.
    pub et_ms: u32,
    /// Output signal.
    pub q: bool,
}

impl Ton {
    /// Creates a new TON instance with given preset time.
    pub const fn new(pt_ms: u32) -> Self {
        Self {
            pt_ms,
            et_ms: 0,
            q: false,
        }
    }

    /// Evaluates the timer for a discrete cycle of `dt_ms` milliseconds.
    pub fn update(&mut self, in_sig: bool, dt_ms: u32) -> bool {
        if in_sig {
            if self.et_ms < self.pt_ms {
                self.et_ms = (self.et_ms + dt_ms).min(self.pt_ms);
            }
            self.q = self.et_ms >= self.pt_ms;
        } else {
            self.et_ms = 0;
            self.q = false;
        }
        self.q
    }
}

/// TOF - Off-Delay Timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tof {
    /// Preset Time in milliseconds.
    pub pt_ms: u32,
    /// Elapsed Time in milliseconds.
    pub et_ms: u32,
    /// Output signal.
    pub q: bool,
}

impl Tof {
    /// Creates a new TOF instance.
    pub const fn new(pt_ms: u32) -> Self {
        Self {
            pt_ms,
            et_ms: 0,
            q: false,
        }
    }

    /// Evaluates the off-delay timer.
    pub fn update(&mut self, in_sig: bool, dt_ms: u32) -> bool {
        if in_sig {
            self.q = true;
            self.et_ms = 0;
        } else if self.q {
            self.et_ms = (self.et_ms + dt_ms).min(self.pt_ms);
            if self.et_ms >= self.pt_ms {
                self.q = false;
            }
        }
        self.q
    }
}

/// TP - Pulse Timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tp {
    /// Preset Pulse Duration in milliseconds.
    pub pt_ms: u32,
    /// Elapsed Time in milliseconds.
    pub et_ms: u32,
    /// Output signal.
    pub q: bool,
    running: bool,
    prev_in: bool,
}

impl Tp {
    /// Creates a new TP pulse timer.
    pub const fn new(pt_ms: u32) -> Self {
        Self {
            pt_ms,
            et_ms: 0,
            q: false,
            running: false,
            prev_in: false,
        }
    }

    /// Evaluates the pulse timer.
    pub fn update(&mut self, in_sig: bool, dt_ms: u32) -> bool {
        let rising_edge = in_sig && !self.prev_in;
        self.prev_in = in_sig;

        if !self.running && rising_edge {
            self.running = true;
            self.q = true;
            self.et_ms = 0;
        } else if self.running {
            self.et_ms = (self.et_ms + dt_ms).min(self.pt_ms);
            if self.et_ms >= self.pt_ms {
                self.running = false;
                self.q = false;
            }
        }

        self.q
    }
}

/// CTU - Count Up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ctu {
    /// Preset Value.
    pub pv: u32,
    /// Current Counter Value.
    pub cv: u32,
    /// Output signal (true when CV >= PV).
    pub q: bool,
    prev_cu: bool,
}

impl Ctu {
    /// Creates a new CTU instance.
    pub const fn new(pv: u32) -> Self {
        Self {
            pv,
            cv: 0,
            q: false,
            prev_cu: false,
        }
    }

    /// Updates the up-counter.
    pub fn update(&mut self, cu: bool, reset: bool) -> bool {
        if reset {
            self.cv = 0;
            self.q = false;
        } else {
            let rising = cu && !self.prev_cu;
            if rising && self.cv < u32::MAX {
                self.cv += 1;
            }
            self.q = self.cv >= self.pv;
        }
        self.prev_cu = cu;
        self.q
    }
}

/// CTD - Count Down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ctd {
    /// Preset Value.
    pub pv: u32,
    /// Current Counter Value.
    pub cv: u32,
    /// Output signal (true when CV == 0).
    pub q: bool,
    prev_cd: bool,
}

impl Ctd {
    /// Creates a new CTD instance.
    pub const fn new(pv: u32) -> Self {
        Self {
            pv,
            cv: pv,
            q: false,
            prev_cd: false,
        }
    }

    /// Updates the down-counter.
    pub fn update(&mut self, cd: bool, load: bool) -> bool {
        if load {
            self.cv = self.pv;
            self.q = self.cv == 0;
        } else {
            let rising = cd && !self.prev_cd;
            if rising && self.cv > 0 {
                self.cv -= 1;
            }
            self.q = self.cv == 0;
        }
        self.prev_cd = cd;
        self.q
    }
}

/// PID_Compact - Standard industrial closed-loop PID controller with anti-windup clamping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidCompact {
    /// Proportional gain.
    pub kp: f32,
    /// Integral time constant (seconds).
    pub ti: f32,
    /// Derivative time constant (seconds).
    pub td: f32,
    /// Minimum output clamp.
    pub out_min: f32,
    /// Maximum output clamp.
    pub out_max: f32,
    /// Accumulated integrator sum.
    integral: f32,
    /// Previous input for derivative calculation.
    prev_input: f32,
    /// Filtered derivative state.
    filtered_derivative: f32,
}

impl Default for PidCompact {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ti: 10.0,
            td: 0.0,
            out_min: 0.0,
            out_max: 100.0,
            integral: 0.0,
            prev_input: 0.0,
            filtered_derivative: 0.0,
        }
    }
}

impl PidCompact {
    /// Creates a new PID_Compact controller.
    pub const fn new(kp: f32, ti: f32, td: f32, out_min: f32, out_max: f32) -> Self {
        Self {
            kp,
            ti,
            td,
            out_min,
            out_max,
            integral: 0.0,
            prev_input: 0.0,
            filtered_derivative: 0.0,
        }
    }

    /// Resets internal integrator and derivative states.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_input = 0.0;
        self.filtered_derivative = 0.0;
    }

    /// Evaluates the PID loop for one cycle.
    ///
    /// `dt`: cycle time in seconds (e.g. 0.01 for 10ms cycle).
    pub fn update(&mut self, setpoint: f32, input: f32, dt: f32) -> f32 {
        let error = setpoint - input;

        // Proportional term
        let p_term = self.kp * error;

        // Integral term with trapezoidal/Euler integration and anti-windup clamping
        if self.ti > 1e-4 {
            let i_delta = (self.kp / self.ti) * error * dt;
            self.integral += i_delta;
            // Anti-windup clamping on integrator
            if self.integral > self.out_max {
                self.integral = self.out_max;
            } else if self.integral < self.out_min {
                self.integral = self.out_min;
            }
        }

        // Derivative term on measurement (derivative kick prevention) with low-pass filter
        let d_term = if self.td > 1e-4 && dt > 1e-6 {
            let derivative_raw = -self.kp * self.td * (input - self.prev_input) / dt;
            // First-order derivative filter with alpha = 0.8
            self.filtered_derivative = 0.8 * self.filtered_derivative + 0.2 * derivative_raw;
            self.filtered_derivative
        } else {
            0.0
        };

        self.prev_input = input;

        // Unclamped output
        let mut output = p_term + self.integral + d_term;

        // Final output saturation clamp
        if output > self.out_max {
            output = self.out_max;
        } else if output < self.out_min {
            output = self.out_min;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ton_timer() {
        let mut ton = Ton::new(100); // 100 ms timer
        assert!(!ton.update(true, 40));
        assert_eq!(ton.et_ms, 40);
        assert!(!ton.update(true, 50));
        assert_eq!(ton.et_ms, 90);
        // Exceeds 100ms
        assert!(ton.update(true, 20));
        assert_eq!(ton.et_ms, 100);
        assert!(ton.q);

        // Falling edge resets timer
        assert!(!ton.update(false, 10));
        assert_eq!(ton.et_ms, 0);
    }

    #[test]
    fn test_ctu_counter() {
        let mut ctu = Ctu::new(3);
        assert!(!ctu.update(true, false)); // 1
        assert!(!ctu.update(false, false));
        assert!(!ctu.update(true, false)); // 2
        assert!(!ctu.update(false, false));
        assert!(ctu.update(true, false)); // 3 -> Q = true
        assert_eq!(ctu.cv, 3);
    }

    #[test]
    fn test_pid_compact() {
        let mut pid = PidCompact::new(2.0, 1.0, 0.0, 0.0, 100.0);
        // Error = 10, Kp = 2 -> P = 20, dt = 0.1 -> I = (2/1) * 10 * 0.1 = 2
        let out = pid.update(10.0, 0.0, 0.1);
        assert!((out - 22.0).abs() < 1e-3);
    }
}
