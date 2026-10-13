//! Vectorized Normalization layers (RMSNorm, LayerNorm).
//! Supports x86_64 (AVX-512, AVX2+FMA) and ARM64 (NEON).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Evaluates Root Mean Square Normalization (RMSNorm):
/// dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
///
/// # Safety
/// Pointers `x`, `weight`, and `dst` must point to valid allocations of length `n`.
pub unsafe fn rms_norm(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    eps: f32,
) {
    if n == 0 {
        return;
    }

    // 1. Calculate sum of squares
    let mut sum_sq = 0.0f32;

    #[cfg(target_arch = "x86_64")]
    {
        if crate::microkernel::is_avx512_supported() && n >= 16 {
            sum_sq = sum_sq_avx512(n, x);
        } else if crate::microkernel::is_avx2_fma_supported() && n >= 8 {
            sum_sq = sum_sq_avx2(n, x);
        } else {
            for i in 0..n {
                let v = *x.add(i);
                sum_sq += v * v;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= 4 {
            sum_sq = sum_sq_neon(n, x);
        } else {
            for i in 0..n {
                let v = *x.add(i);
                sum_sq += v * v;
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for i in 0..n {
            let v = *x.add(i);
            sum_sq += v * v;
        }
    }

    let mean_sq = sum_sq / (n as f32);
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();

    // 2. Normalize and scale by weight
    #[cfg(target_arch = "x86_64")]
    {
        if crate::microkernel::is_avx512_supported() && n >= 16 {
            normalize_and_scale_avx512(n, x, weight, dst, inv_rms);
            return;
        }
        if crate::microkernel::is_avx2_fma_supported() && n >= 8 {
            normalize_and_scale_avx2(n, x, weight, dst, inv_rms);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= 4 {
            normalize_and_scale_neon(n, x, weight, dst, inv_rms);
            return;
        }
    }

    for i in 0..n {
        *dst.add(i) = (*x.add(i) * inv_rms) * *weight.add(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_sq_avx512(n: usize, x: *const f32) -> f32 {
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let vx = _mm512_loadu_ps(x.add(i));
        acc = _mm512_fmadd_ps(vx, vx, acc);
        i += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc);
    while i < n {
        let v = *x.add(i);
        total += v * v;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn normalize_and_scale_avx512(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    inv_rms: f32,
) {
    let v_scale = _mm512_set1_ps(inv_rms);
    let mut i = 0;

    while i + 16 <= n {
        let vx = _mm512_loadu_ps(x.add(i));
        let vw = _mm512_loadu_ps(weight.add(i));
        let v_norm = _mm512_mul_ps(vx, v_scale);
        let v_out = _mm512_mul_ps(v_norm, vw);
        _mm512_storeu_ps(dst.add(i), v_out);
        i += 16;
    }

    while i < n {
        *dst.add(i) = (*x.add(i) * inv_rms) * *weight.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_neon(n: usize, x: *const f32) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= n {
        let vx = vld1q_f32(x.add(i));
        acc = vfmaq_f32(acc, vx, vx);
        i += 4;
    }

    let mut total = vaddvq_f32(acc);
    while i < n {
        let v = *x.add(i);
        total += v * v;
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_and_scale_neon(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    inv_rms: f32,
) {
    let v_scale = vdupq_n_f32(inv_rms);
    let mut i = 0;

    while i + 4 <= n {
        let vx = vld1q_f32(x.add(i));
        let vw = vld1q_f32(weight.add(i));
        let v_norm = vmulq_f32(vx, v_scale);
        let v_out = vmulq_f32(v_norm, vw);
        vst1q_f32(dst.add(i), v_out);
        i += 4;
    }

    while i < n {
        *dst.add(i) = (*x.add(i) * inv_rms) * *weight.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_avx2(n: usize, x: *const f32) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let vx0 = _mm256_loadu_ps(x.add(i));
        acc0 = _mm256_fmadd_ps(vx0, vx0, acc0);

        let vx1 = _mm256_loadu_ps(x.add(i + 8));
        acc1 = _mm256_fmadd_ps(vx1, vx1, acc1);

        i += 16;
    }

    if i + 8 <= n {
        let vx = _mm256_loadu_ps(x.add(i));
        acc0 = _mm256_fmadd_ps(vx, vx, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while i < n {
        let v = *x.add(i);
        total += v * v;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn normalize_and_scale_avx2(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    inv_rms: f32,
) {
    let v_scale = _mm256_set1_ps(inv_rms);
    let mut i = 0;

    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.add(i));
        let vw = _mm256_loadu_ps(weight.add(i));
        let v_norm = _mm256_mul_ps(vx, v_scale);
        let v_out = _mm256_mul_ps(v_norm, vw);
        _mm256_storeu_ps(dst.add(i), v_out);
        i += 8;
    }

    while i < n {
        *dst.add(i) = (*x.add(i) * inv_rms) * *weight.add(i);
        i += 1;
    }
}
