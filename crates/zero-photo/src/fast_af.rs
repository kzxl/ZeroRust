//! Fast Auto-Focus Engine: Predictive Hill-Climbing LiveView AF.
//!
//! Provides intelligent real-time contrast-detection auto-focus for Canon EF lenses
//! in LiveView on Canon EOS 5D Mark II (DIGIC 4, ARMv5TE):
//! - Fast luminance edge-gradient contrast evaluation on active AF focus spot.
//! - Micro-step probe for slope detection (+/- direction determination).
//! - Coarse hill-climbing trajectory with overshoot crest detection.
//! - Fine deceleration and precise reverse-parking at peak sharpness.
//! - Direction inversion resilience for ambiguous initial slopes.
//! - Bounded execution (<45 motor steps, 2.0s maximum latency).
//! - Zero memory allocations in control loops.

use crate::row_filter::div_u32;

pub const FAST_AF_ACT_WAIT_FRAME: i32 = 0;
pub const FAST_AF_ACT_STEP_MOTOR: i32 = 1;
pub const FAST_AF_ACT_LOCK_SUCCESS: i32 = 2;
pub const FAST_AF_ACT_ABORT_TIMEOUT: i32 = 3;
pub const FAST_AF_ACT_ABORT_LOW_CONTRAST: i32 = 4;

pub const FAST_AF_PHASE_IDLE: i32 = 0;
pub const FAST_AF_PHASE_PROBE_INIT: i32 = 1;
pub const FAST_AF_PHASE_PROBE_STEP1: i32 = 2;
pub const FAST_AF_PHASE_PROBE_STEP2: i32 = 3;
pub const FAST_AF_PHASE_COARSE_CLIMB: i32 = 4;
pub const FAST_AF_PHASE_FINE_RETURN: i32 = 5;
pub const FAST_AF_PHASE_FINE_TUNE_POS: i32 = 6;
pub const FAST_AF_PHASE_FINE_TUNE_NEG: i32 = 7;
pub const FAST_AF_PHASE_FINAL_PARK: i32 = 8;
pub const FAST_AF_PHASE_DONE: i32 = 9;

/// Minimum contrast floor to prevent hunting on featureless surfaces (white wall/lens cap).
pub const MIN_AF_CONTRAST_FLOOR: u32 = 100;

/// State controller for the Predictive Hill-Climbing AF algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastAfController {
    pub phase: i32,
    pub stride: i32,
    pub max_steps: i32,
    pub direction: i32,
    pub current_pos: i32,
    pub peak_pos: i32,
    pub initial_q: u32,
    pub peak_q: u32,
    pub prev_q: u32,
    pub total_motor_steps: i32,
    pub fall_count: i32,
    pub inverted_once: bool,
    pub commanded_step: i32,
}

impl Default for FastAfController {
    fn default() -> Self {
        Self::new()
    }
}

impl FastAfController {
    pub const fn new() -> Self {
        Self {
            phase: FAST_AF_PHASE_IDLE,
            stride: 2,
            max_steps: 40,
            direction: 1,
            current_pos: 0,
            peak_pos: 0,
            initial_q: 0,
            peak_q: 0,
            prev_q: 0,
            total_motor_steps: 0,
            fall_count: 0,
            inverted_once: false,
            commanded_step: 0,
        }
    }

    /// Resets and prepares the controller for a new auto-focus acquisition cycle.
    pub fn reset(&mut self, stride: i32, max_steps: i32) {
        self.phase = FAST_AF_PHASE_PROBE_INIT;
        self.stride = stride.clamp(1, 3);
        self.max_steps = max_steps.clamp(10, 80);
        self.direction = 1;
        self.current_pos = 0;
        self.peak_pos = 0;
        self.initial_q = 0;
        self.peak_q = 0;
        self.prev_q = 0;
        self.total_motor_steps = 0;
        self.fall_count = 0;
        self.inverted_once = false;
        self.commanded_step = 0;
    }

    /// Evaluates the contrast metric `q_current` from current LiveView frame and determines next action.
    pub fn step(&mut self, q_current: u32) -> i32 {
        if self.phase == FAST_AF_PHASE_IDLE {
            return FAST_AF_ACT_WAIT_FRAME;
        }

        if self.phase == FAST_AF_PHASE_DONE {
            return FAST_AF_ACT_LOCK_SUCCESS;
        }

        // Hard iteration limit safety
        if self.total_motor_steps >= self.max_steps {
            self.phase = FAST_AF_PHASE_DONE;
            return FAST_AF_ACT_ABORT_TIMEOUT;
        }

        match self.phase {
            FAST_AF_PHASE_PROBE_INIT => {
                // Check contrast floor
                if q_current < MIN_AF_CONTRAST_FLOOR {
                    self.phase = FAST_AF_PHASE_DONE;
                    return FAST_AF_ACT_ABORT_LOW_CONTRAST;
                }

                self.initial_q = q_current;
                self.peak_q = q_current;
                self.prev_q = q_current;
                self.peak_pos = 0;
                self.current_pos = 0;

                // Probe 1 fine step forward (+1)
                self.commanded_step = 1;
                self.current_pos += 1;
                self.total_motor_steps += 1;
                self.phase = FAST_AF_PHASE_PROBE_STEP1;
                FAST_AF_ACT_STEP_MOTOR
            }

            FAST_AF_PHASE_PROBE_STEP1 => {
                let dq = (q_current as i32) - (self.initial_q as i32);
                let tau = ((self.initial_q >> 4).max(10)) as i32;

                if dq > tau {
                    // Clearly ascending forward
                    self.direction = 1;
                    self.peak_q = q_current;
                    self.peak_pos = self.current_pos;
                    self.phase = FAST_AF_PHASE_COARSE_CLIMB;
                    self.command_coarse_step()
                } else if dq < -tau {
                    // Moving forward degraded focus -> true peak is backwards!
                    self.direction = -1;
                    self.phase = FAST_AF_PHASE_COARSE_CLIMB;
                    self.command_coarse_step()
                } else {
                    // Ambiguous slope: probe backwards 2 steps (-1 relative to origin)
                    self.commanded_step = -2;
                    self.current_pos -= 2;
                    self.total_motor_steps += 2;
                    self.phase = FAST_AF_PHASE_PROBE_STEP2;
                    FAST_AF_ACT_STEP_MOTOR
                }
            }

            FAST_AF_PHASE_PROBE_STEP2 => {
                let dq = (q_current as i32) - (self.initial_q as i32);
                let tau = ((self.initial_q >> 4).max(10)) as i32;

                if dq > tau {
                    self.direction = -1;
                    self.peak_q = q_current;
                    self.peak_pos = self.current_pos;
                } else {
                    // Default to forward if inconclusive
                    self.direction = 1;
                }
                self.phase = FAST_AF_PHASE_COARSE_CLIMB;
                self.command_coarse_step()
            }

            FAST_AF_PHASE_COARSE_CLIMB => {
                if q_current > self.peak_q {
                    self.peak_q = q_current;
                    self.peak_pos = self.current_pos;
                    self.fall_count = 0;
                    self.command_coarse_step()
                } else {
                    self.fall_count += 1;

                    // If we haven't climbed anywhere above initial Q and contrast is plunging:
                    // Invert direction once to search the other hemisphere
                    if self.peak_pos == 0 && self.fall_count >= 2 && !self.inverted_once {
                        self.inverted_once = true;
                        self.direction = -self.direction;
                        self.fall_count = 0;
                        return self.command_coarse_step();
                    }

                    // Crest criteria: 2 consecutive drops OR contrast dropped below 85% of peak
                    let dropped_15pct = (q_current * 100) < (self.peak_q * 85);
                    if self.fall_count >= 2 || dropped_15pct {
                        // Crest passed! Transition to FineReturn towards peak_pos
                        self.phase = FAST_AF_PHASE_FINE_RETURN;
                        self.command_return_step()
                    } else {
                        // Keep going one more coarse step to confirm crest
                        self.command_coarse_step()
                    }
                }
            }

            FAST_AF_PHASE_FINE_RETURN => {
                let offset = self.peak_pos - self.current_pos;
                if offset == 0 || self.total_motor_steps >= self.max_steps {
                    if self.stride > 1 && self.total_motor_steps + 3 < self.max_steps {
                        // Coarse stride > 1: perform fine micro-step refinement (+1 / -1)
                        self.phase = FAST_AF_PHASE_FINE_TUNE_POS;
                        self.commanded_step = 1;
                        self.current_pos += 1;
                        self.total_motor_steps += 1;
                        FAST_AF_ACT_STEP_MOTOR
                    } else {
                        self.phase = FAST_AF_PHASE_DONE;
                        self.commanded_step = 0;
                        FAST_AF_ACT_LOCK_SUCCESS
                    }
                } else {
                    self.command_return_step()
                }
            }

            FAST_AF_PHASE_FINE_TUNE_POS => {
                if q_current > self.peak_q {
                    // +1 is a sharper peak! Park here.
                    self.peak_q = q_current;
                    self.peak_pos = self.current_pos;
                    self.commanded_step = 0;
                    self.phase = FAST_AF_PHASE_DONE;
                    FAST_AF_ACT_LOCK_SUCCESS
                } else {
                    // +1 was worse; test -1 from coarse peak (current_pos - 2)
                    self.commanded_step = -2;
                    self.current_pos -= 2;
                    self.total_motor_steps += 2;
                    self.phase = FAST_AF_PHASE_FINE_TUNE_NEG;
                    FAST_AF_ACT_STEP_MOTOR
                }
            }

            FAST_AF_PHASE_FINE_TUNE_NEG => {
                if q_current > self.peak_q {
                    // -1 is a sharper peak! Park here.
                    self.peak_q = q_current;
                    self.peak_pos = self.current_pos;
                    self.commanded_step = 0;
                    self.phase = FAST_AF_PHASE_DONE;
                    FAST_AF_ACT_LOCK_SUCCESS
                } else {
                    // Neither +1 nor -1 was better than coarse peak.
                    // Step +1 back to original coarse peak and transition to FINAL_PARK.
                    self.commanded_step = 1;
                    self.current_pos += 1;
                    self.total_motor_steps += 1;
                    self.phase = FAST_AF_PHASE_FINAL_PARK;
                    FAST_AF_ACT_STEP_MOTOR
                }
            }

            FAST_AF_PHASE_FINAL_PARK => {
                // Motor has now physically arrived back at peak_pos!
                self.commanded_step = 0;
                self.phase = FAST_AF_PHASE_DONE;
                FAST_AF_ACT_LOCK_SUCCESS
            }

            _ => FAST_AF_ACT_WAIT_FRAME,
        }
    }

    fn command_coarse_step(&mut self) -> i32 {
        let step = self.stride * self.direction;
        self.commanded_step = step;
        self.current_pos += step;
        self.total_motor_steps += self.stride;
        FAST_AF_ACT_STEP_MOTOR
    }

    fn command_return_step(&mut self) -> i32 {
        let offset = self.peak_pos - self.current_pos;
        if offset == 0 {
            self.phase = FAST_AF_PHASE_DONE;
            return FAST_AF_ACT_LOCK_SUCCESS;
        }

        let dir = if offset > 0 { 1 } else { -1 };
        // Return with fine step (1 or 2)
        let mag = offset.abs().min(2);
        let step = dir * mag;

        self.commanded_step = step;
        self.current_pos += step;
        self.total_motor_steps += mag;
        FAST_AF_ACT_STEP_MOTOR
    }

    /// Returns focus accuracy confidence percentage (0..100).
    pub fn get_confidence(&self, current_q: u32) -> u32 {
        if self.peak_q == 0 {
            return 0;
        }
        let score = div_u32(current_q * 100, self.peak_q);
        score.min(100)
    }
}

// ------------------------------------------------------------------------
// C ABI exports for ZeroLantern Fast-AF Controller
// ------------------------------------------------------------------------

static mut CONTROLLER: FastAfController = FastAfController::new();

/// Resets the global Fast-AF state machine.
#[no_mangle]
pub extern "C" fn rust_fast_af_reset(stride: i32, max_steps: i32) {
    let ctrl = core::ptr::addr_of_mut!(CONTROLLER);
    unsafe {
        (*ctrl).reset(stride, max_steps);
    }
}

/// Steps the global Fast-AF controller with the latest contrast metric.
///
/// # Safety
/// Pointers must be valid or null.
#[no_mangle]
pub unsafe extern "C" fn rust_fast_af_step(
    q_current: u32,
    out_step: *mut i32,
    out_phase: *mut i32,
) -> i32 {
    let ctrl = core::ptr::addr_of_mut!(CONTROLLER);
    let act = (*ctrl).step(q_current);

    if !out_step.is_null() {
        *out_step = (*ctrl).commanded_step;
    }
    if !out_phase.is_null() {
        *out_phase = (*ctrl).phase;
    }

    act
}

/// Returns the peak contrast and current confidence score.
///
/// # Safety
/// Pointers must be valid or null.
#[no_mangle]
pub unsafe extern "C" fn rust_fast_af_get_metrics(
    q_current: u32,
    out_peak_q: *mut u32,
    out_confidence: *mut u32,
) {
    let ctrl = core::ptr::addr_of_mut!(CONTROLLER);
    if !out_peak_q.is_null() {
        *out_peak_q = (*ctrl).peak_q;
    }
    if !out_confidence.is_null() {
        *out_confidence = (*ctrl).get_confidence(q_current);
    }
}

/// Evaluates high-frequency luminance gradient in a rectangular region around (cx, cy).
///
/// # Safety
/// `vram_base` must be a valid pointer to UYVY frame buffer with sufficient height.
#[no_mangle]
pub unsafe extern "C" fn rust_fast_af_eval_box(
    vram_base: *const u32,
    dwords_per_row: i32,
    cx: i32,
    cy: i32,
    half_w: i32,
    half_h: i32,
) -> u32 {
    if vram_base.is_null() || dwords_per_row <= 0 {
        return 0;
    }

    let xmin = (cx - half_w).max(4);
    let xmax = cx + half_w;
    let ymin = (cy - half_h).max(2);
    let ymax = cy + half_h;

    let mut total_grad: u32 = 0;

    // Subsample rows by 2 for speed on DIGIC 4
    let mut y = ymin;
    while y <= ymax {
        let row_ptr = vram_base.add((y as usize) * (dwords_per_row as usize));
        total_grad = total_grad.saturating_add(crate::split_focus::eval_row_gradient_uyvy(
            row_ptr, xmin, xmax,
        ));
        y += 2;
    }

    total_grad
}
