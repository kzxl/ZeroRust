//! Digital Split-Image Rangefinder & Focus Peaking Engine for LiveView.
//!
//! Emulates the classic optical split-image focusing screen (Canon F-1, Leica R, Nikon FM2)
//! directly in real-time YUV422 LiveView:
//! - Central circular reticle with a horizontal split-line dividing top and bottom halves.
//! - Horizontal shear displacement of the bottom half proportional to defocus.
//! - Real-time edge gradient accumulation (1D Sobel/Laplacian proxy).
//! - Automatic focus lock detection (Green Lock Confirmation + Score 0..100).
//! - Zero memory allocations in frame rendering loops.

/// Maximum radius for the circular rangefinder reticle (pixels).
pub const SPLIT_RETICLE_RADIUS: i32 = 56;

/// Precomputed integer circle radius lookup table for R = 56: dx = sqrt(56^2 - dy^2).
/// Eliminates all runtime square-root / floating-point calculations on ARMv5TE.
pub static CIRCLE_DX_56: [u8; 57] = [
    56, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    55, 54, 54, 54, 54, 53, 53, 53, 52, 52,
    52, 51, 51, 50, 50, 49, 49, 48, 47, 47,
    46, 45, 45, 44, 43, 42, 41, 40, 39, 38,
    37, 36, 35, 33, 32, 31, 29, 27, 26, 24,
    22, 19, 17, 14, 10, 6, 0,
];

/// State tracker for the digital rangefinder focus evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitFocusState {
    pub q_smooth: u32,
    pub q_peak: u32,
    pub focus_score: u32,
    pub is_locked: bool,
    pub shear_offset: i32,
}

impl Default for SplitFocusState {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitFocusState {
    pub const fn new() -> Self {
        Self {
            q_smooth: 0,
            q_peak: 2000, // baseline floor to prevent division by zero / wild jumps
            focus_score: 0,
            is_locked: false,
            shear_offset: 12,
        }
    }

    /// Updates focus state from a new raw gradient sum measured in the central reticle.
    ///
    /// Computes exponential moving average, updates decaying peak tracker,
    /// determines focus lock (score >= 90%), and calculates shear displacement.
    pub fn update(&mut self, q_raw: u32) {
        // Fast 1-pole IIR smoothing: (7 * q_smooth + q_raw) / 8
        self.q_smooth = ((self.q_smooth * 7) + q_raw) >> 3;

        // Peak tracking with slow decay (decay by 1/64 every frame ~ 1.5% decay per frame)
        if self.q_smooth > self.q_peak {
            self.q_peak = self.q_smooth;
        } else {
            let decay = (self.q_peak >> 6).max(1);
            if self.q_peak > 2000 + decay {
                self.q_peak -= decay;
            }
        }

        // Focus score ratio: (q_smooth * 100) / q_peak
        if self.q_peak > 0 {
            let score = crate::row_filter::div_u32(self.q_smooth * 100, self.q_peak);
            self.focus_score = score.min(100);
        } else {
            self.focus_score = 0;
        }

        // Focus lock threshold: >= 88% of local peak sharpness
        self.is_locked = self.focus_score >= 88;

        // Shear displacement: out of focus -> up to 12 px shear; in focus -> 0 px
        if self.is_locked {
            self.shear_offset = 0;
        } else {
            let def = 88 - self.focus_score;
            // Shear in steps of 2 pixels (even number for UYVY pixel pair alignment)
            let s = (crate::row_filter::div_u32(def * 12, 88) as i32) & !1;
            self.shear_offset = s.clamp(0, 14);
        }
    }
}

/// Evaluates high-frequency edge gradient on a single UYVY row within the range [xmin, xmax].
///
/// Each UYVY DWORD in Canon VRAM contains:
/// - Byte 0: U
/// - Byte 1: Y0
/// - Byte 2: V
/// - Byte 3: Y1
///
/// # Safety
/// `row_src` must be valid for reads of at least `(xmax / 2) + 1` DWORDs and 4-byte aligned.
pub unsafe fn eval_row_gradient_uyvy(row_src: *const u32, xmin: i32, xmax: i32) -> u32 {
    if row_src.is_null() || xmax <= xmin + 2 {
        return 0;
    }

    let start_pair = ((xmin >> 1).max(1)) as usize;
    let end_pair = (xmax >> 1) as usize;
    let mut sum_grad: u32 = 0;

    for i in start_pair..end_pair {
        let p_prev = *row_src.add(i - 1);
        let p_curr = *row_src.add(i);

        let y_prev1 = (p_prev >> 24) & 0xFF;
        let y_curr0 = (p_curr >> 8) & 0xFF;
        let y_curr1 = (p_curr >> 24) & 0xFF;

        let d0 = (y_curr0 as i32 - y_prev1 as i32).abs() as u32;
        let d1 = (y_curr1 as i32 - y_curr0 as i32).abs() as u32;

        sum_grad += d0 + d1;
    }

    sum_grad
}

// ------------------------------------------------------------------------
// C ABI exports for ZeroLantern LiveView display engine
// ------------------------------------------------------------------------

/// Computes half-width of reticle at offset `dy` from center (0 <= dy <= 56).
#[no_mangle]
pub extern "C" fn rust_split_get_circle_dx(dy: i32) -> i32 {
    let abs_dy = dy.unsigned_abs() as usize;
    if abs_dy >= CIRCLE_DX_56.len() {
        0
    } else {
        CIRCLE_DX_56[abs_dy] as i32
    }
}

/// Updates split focus state from current frame gradient sum and returns results.
///
/// # Safety
/// Pointers must be valid or null.
#[no_mangle]
pub unsafe extern "C" fn rust_split_update_state(
    q_raw: u32,
    out_shear: *mut i32,
    out_score: *mut u32,
    out_locked: *mut i32,
) {
    // Static state on embedded runtime (thread-safe on single-core DryOS display CBR)
    static mut STATE: SplitFocusState = SplitFocusState::new();
    let state_ptr = core::ptr::addr_of_mut!(STATE);

    (*state_ptr).update(q_raw);

    if !out_shear.is_null() {
        *out_shear = (*state_ptr).shear_offset;
    }
    if !out_score.is_null() {
        *out_score = (*state_ptr).focus_score;
    }
    if !out_locked.is_null() {
        *out_locked = if (*state_ptr).is_locked { 1 } else { 0 };
    }
}
