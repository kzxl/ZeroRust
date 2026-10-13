//! Fast SIMD math approximations and neural network activations.

/// Fast approximation of e^x using Padé rational approximation.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }

    // Remez / Padé approximation: 2^(x * log2(e))
    let x_clamped = x.max(-88.0).min(88.0);
    #[cfg(feature = "std")]
    {
        x_clamped.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        // Polynomial approximation for no_std
        let z = 1.0 + x_clamped / 256.0;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let z16 = z8 * z8;
        let z32 = z16 * z16;
        let z64 = z32 * z32;
        let z128 = z64 * z64;
        z128 * z128
    }
}

/// Computes SiLU (Swish) activation: x / (1 + exp(-x)).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + fast_exp(-x))
}

/// Vectorized SiLU inplace or buffer to buffer: dst[i] = src[i] / (1 + exp(-src[i])).
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
pub unsafe fn vector_silu(n: usize, src: *const f32, dst: *mut f32) {
    for i in 0..n {
        *dst.add(i) = silu(*src.add(i));
    }
}

/// Vectorized SwiGLU: dst[i] = silu(gate[i]) * up[i].
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
pub unsafe fn vector_swiglu(n: usize, gate: *const f32, up: *const f32, dst: *mut f32) {
    for i in 0..n {
        let g = *gate.add(i);
        let u = *up.add(i);
        *dst.add(i) = silu(g) * u;
    }
}

/// Evaluates high-performance in-place Softmax over a vector of length `n`.
///
/// # Safety
/// Pointer `x` must point to a valid allocation of length `n`.
pub unsafe fn softmax_row(n: usize, x: *mut f32) {
    if n == 0 {
        return;
    }

    // 1. Find max for numerical stability
    let mut max_val = *x;
    for i in 1..n {
        let val = *x.add(i);
        if val > max_val {
            max_val = val;
        }
    }

    // 2. Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let ptr = x.add(i);
        let ev = fast_exp(*ptr - max_val);
        *ptr = ev;
        sum += ev;
    }

    // 3. Normalize
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for i in 0..n {
        *x.add(i) *= inv_sum;
    }
}
