//! Hardware-intrinsic SIMD GEMM and dot-product microkernels.
//! Supports x86_64 (AVX-512, AVX2+FMA) and ARM64 (NEON).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Checks whether AVX-512F hardware SIMD acceleration is supported on the running CPU.
#[inline(always)]
pub fn is_avx512_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx512f") {
                return true;
            }
        }
    }
    false
}

/// Checks whether AVX2 and FMA hardware SIMD acceleration is supported on the running CPU.
#[inline(always)]
pub fn is_avx2_fma_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return true;
            }
        }
    }
    false
}

/// Checks whether ARM64 NEON hardware SIMD acceleration is supported on the running CPU.
#[inline(always)]
pub fn is_neon_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return true;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Evaluates high-performance FP32 vector dot product: sum(a[i] * b[i]).
///
/// # Safety
/// Pointers `a` and `b` must point to valid allocations of length `n`.
pub unsafe fn dot_f32(n: usize, a: *const f32, b: *const f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() && n >= 16 {
            return dot_f32_avx512(n, a, b);
        }
        if is_avx2_fma_supported() && n >= 8 {
            return dot_f32_avx2(n, a, b);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= 4 {
            return dot_f32_neon(n, a, b);
        }
    }

    let mut sum = 0.0f32;
    for i in 0..n {
        sum += *a.add(i) * *b.add(i);
    }
    sum
}

/// Evaluates high-performance Int8 vector dot product with Int32 accumulation: sum(a[i] * b[i]).
///
/// # Safety
/// Pointers `a` and `b` must point to valid allocations of length `n`.
pub unsafe fn dot_i8(n: usize, a: *const i8, b: *const i8) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_supported() && n >= 16 {
            return dot_i8_avx2(n, a, b);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n >= 8 {
            return dot_i8_neon(n, a, b);
        }
    }

    let mut sum = 0i32;
    for i in 0..n {
        sum += (*a.add(i) as i32) * (*b.add(i) as i32);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(n: usize, a: *const i8, b: *const i8) -> i32 {
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;

    while i + 16 <= n {
        let va8 = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb8 = _mm_loadu_si128(b.add(i) as *const __m128i);
        let va16 = _mm256_cvtepi8_epi16(va8);
        let vb16 = _mm256_cvtepi8_epi16(vb8);
        // _mm256_madd_epi16 multiplies adjacent i16 pairs and sums them into i32
        let prod32 = _mm256_madd_epi16(va16, vb16);
        acc = _mm256_add_epi32(acc, prod32);
        i += 16;
    }

    let mut buf = [0i32; 8];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    let mut total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while i < n {
        total += (*a.add(i) as i32) * (*b.add(i) as i32);
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(n: usize, a: *const i8, b: *const i8) -> i32 {
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;

    while i + 8 <= n {
        let va = vld1_s8(a.add(i));
        let vb = vld1_s8(b.add(i));
        let prod16 = vmull_s8(va, vb);
        acc = vpadalq_s16(acc, prod16);
        i += 8;
    }

    let mut total = vaddvq_s32(acc);
    while i < n {
        total += (*a.add(i) as i32) * (*b.add(i) as i32);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(n: usize, a: *const f32, b: *const f32) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 32 <= n {
        let va0 = _mm512_loadu_ps(a.add(i));
        let vb0 = _mm512_loadu_ps(b.add(i));
        acc0 = _mm512_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm512_loadu_ps(a.add(i + 16));
        let vb1 = _mm512_loadu_ps(b.add(i + 16));
        acc1 = _mm512_fmadd_ps(va1, vb1, acc1);

        i += 32;
    }

    if i + 16 <= n {
        let va = _mm512_loadu_ps(a.add(i));
        let vb = _mm512_loadu_ps(b.add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }

    let acc = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(acc);

    while i < n {
        total += *a.add(i) * *b.add(i);
        i += 1;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(n: usize, a: *const f32, b: *const f32) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 8 <= n {
        let va0 = vld1q_f32(a.add(i));
        let vb0 = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(a.add(i + 4));
        let vb1 = vld1q_f32(b.add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);

        i += 8;
    }

    if i + 4 <= n {
        let va = vld1q_f32(a.add(i));
        let vb = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }

    let acc = vaddq_f32(acc0, acc1);
    let mut total = vaddvq_f32(acc);

    while i < n {
        total += *a.add(i) * *b.add(i);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(n: usize, a: *const f32, b: *const f32) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        i += 16;
    }

    if i + 8 <= n {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while i < n {
        total += *a.add(i) * *b.add(i);
        i += 1;
    }

    total
}

/// Evaluates C = alpha * (A @ B) + beta * C for row range [m_start..m_end].
///
/// # Safety
/// Pointers `a`, `b`, and `c` must point to valid allocations satisfying the specified dimensions and strides.
pub unsafe fn gemm_serial_range(
    m_start: usize,
    m_end: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if csb == 1 && csc == 1 && is_avx2_fma_supported() {
            gemm_range_avx2_fma(
                m_start, m_end, n, k, alpha, a, rsa, csa, b, rsb, beta, c, rsc,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if csb == 1 && csc == 1 {
            gemm_range_neon(
                m_start, m_end, n, k, alpha, a, rsa, csa, b, rsb, beta, c, rsc,
            );
            return;
        }
    }

    // Scalar fallback
    for i in m_start..m_end {
        let p_a = a.offset(i as isize * rsa);
        let p_c = c.offset(i as isize * rsc);

        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *p_a.offset(p as isize * csa) * *b.offset(p as isize * rsb + j as isize * csb);
            }
            let ptr_c = p_c.offset(j as isize * csc);
            if beta == 0.0 {
                *ptr_c = alpha * sum;
            } else {
                *ptr_c = alpha * sum + beta * *ptr_c;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_range_neon(
    m_start: usize,
    m_end: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
) {
    for i in m_start..m_end {
        let p_a = a.offset(i as isize * rsa);
        let p_c = c.offset(i as isize * rsc);
        let mut j = 0;

        while j + 8 <= n {
            let mut c0 = vdupq_n_f32(0.0);
            let mut c1 = vdupq_n_f32(0.0);

            for p in 0..k {
                let p_bp = b.offset(p as isize * rsb + j as isize);
                let vb0 = vld1q_f32(p_bp);
                let vb1 = vld1q_f32(p_bp.add(4));
                let va = vdupq_n_f32(*p_a.offset(p as isize * csa));
                c0 = vfmaq_f32(c0, va, vb0);
                c1 = vfmaq_f32(c1, va, vb1);
            }

            if alpha != 1.0 {
                let v_alpha = vdupq_n_f32(alpha);
                c0 = vmulq_f32(c0, v_alpha);
                c1 = vmulq_f32(c1, v_alpha);
            }

            if beta == 0.0 {
                vst1q_f32(p_c.add(j), c0);
                vst1q_f32(p_c.add(j + 4), c1);
            } else {
                let v_beta = vdupq_n_f32(beta);
                let orig0 = vmulq_f32(vld1q_f32(p_c.add(j)), v_beta);
                let orig1 = vmulq_f32(vld1q_f32(p_c.add(j + 4)), v_beta);
                vst1q_f32(p_c.add(j), vaddq_f32(c0, orig0));
                vst1q_f32(p_c.add(j + 4), vaddq_f32(c1, orig1));
            }
            j += 8;
        }

        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *p_a.offset(p as isize * csa) * *b.offset(p as isize * rsb + j as isize);
            }
            update_c(p_c.add(j), sum, alpha, beta);
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_range_avx2_fma(
    m_start: usize,
    m_end: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
) {
    let mut i = m_start;

    while i + 4 <= m_end {
        let p_a0 = a.offset((i as isize) * rsa);
        let p_a1 = a.offset((i as isize + 1) * rsa);
        let p_a2 = a.offset((i as isize + 2) * rsa);
        let p_a3 = a.offset((i as isize + 3) * rsa);

        let p_c0 = c.offset((i as isize) * rsc);
        let p_c1 = c.offset((i as isize + 1) * rsc);
        let p_c2 = c.offset((i as isize + 2) * rsc);
        let p_c3 = c.offset((i as isize + 3) * rsc);

        let mut j = 0;
        while j + 16 <= n {
            let mut c00 = _mm256_setzero_ps();
            let mut c01 = _mm256_setzero_ps();
            let mut c10 = _mm256_setzero_ps();
            let mut c11 = _mm256_setzero_ps();
            let mut c20 = _mm256_setzero_ps();
            let mut c21 = _mm256_setzero_ps();
            let mut c30 = _mm256_setzero_ps();
            let mut c31 = _mm256_setzero_ps();

            for p in 0..k {
                let p_bp = b.offset(p as isize * rsb + j as isize);
                let vb0 = _mm256_loadu_ps(p_bp);
                let vb1 = _mm256_loadu_ps(p_bp.add(8));

                let va0 = _mm256_set1_ps(*p_a0.offset(p as isize * csa));
                c00 = _mm256_fmadd_ps(va0, vb0, c00);
                c01 = _mm256_fmadd_ps(va0, vb1, c01);

                let va1 = _mm256_set1_ps(*p_a1.offset(p as isize * csa));
                c10 = _mm256_fmadd_ps(va1, vb0, c10);
                c11 = _mm256_fmadd_ps(va1, vb1, c11);

                let va2 = _mm256_set1_ps(*p_a2.offset(p as isize * csa));
                c20 = _mm256_fmadd_ps(va2, vb0, c20);
                c21 = _mm256_fmadd_ps(va2, vb1, c21);

                let va3 = _mm256_set1_ps(*p_a3.offset(p as isize * csa));
                c30 = _mm256_fmadd_ps(va3, vb0, c30);
                c31 = _mm256_fmadd_ps(va3, vb1, c31);
            }

            if alpha != 1.0 {
                let v_alpha = _mm256_set1_ps(alpha);
                c00 = _mm256_mul_ps(c00, v_alpha);
                c01 = _mm256_mul_ps(c01, v_alpha);
                c10 = _mm256_mul_ps(c10, v_alpha);
                c11 = _mm256_mul_ps(c11, v_alpha);
                c20 = _mm256_mul_ps(c20, v_alpha);
                c21 = _mm256_mul_ps(c21, v_alpha);
                c30 = _mm256_mul_ps(c30, v_alpha);
                c31 = _mm256_mul_ps(c31, v_alpha);
            }

            store_c_tile(p_c0.add(j), c00, c01, beta);
            store_c_tile(p_c1.add(j), c10, c11, beta);
            store_c_tile(p_c2.add(j), c20, c21, beta);
            store_c_tile(p_c3.add(j), c30, c31, beta);

            j += 16;
        }

        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();

            for p in 0..k {
                let p_bp = b.offset(p as isize * rsb + j as isize);
                let vb = _mm256_loadu_ps(p_bp);

                let va0 = _mm256_set1_ps(*p_a0.offset(p as isize * csa));
                c0 = _mm256_fmadd_ps(va0, vb, c0);

                let va1 = _mm256_set1_ps(*p_a1.offset(p as isize * csa));
                c1 = _mm256_fmadd_ps(va1, vb, c1);

                let va2 = _mm256_set1_ps(*p_a2.offset(p as isize * csa));
                c2 = _mm256_fmadd_ps(va2, vb, c2);

                let va3 = _mm256_set1_ps(*p_a3.offset(p as isize * csa));
                c3 = _mm256_fmadd_ps(va3, vb, c3);
            }

            if alpha != 1.0 {
                let v_alpha = _mm256_set1_ps(alpha);
                c0 = _mm256_mul_ps(c0, v_alpha);
                c1 = _mm256_mul_ps(c1, v_alpha);
                c2 = _mm256_mul_ps(c2, v_alpha);
                c3 = _mm256_mul_ps(c3, v_alpha);
            }

            store_c_vec(p_c0.add(j), c0, beta);
            store_c_vec(p_c1.add(j), c1, beta);
            store_c_vec(p_c2.add(j), c2, beta);
            store_c_vec(p_c3.add(j), c3, beta);

            j += 8;
        }

        while j < n {
            scalar_col_4(
                p_a0, p_a1, p_a2, p_a3, csa, b, rsb, j, k, alpha, beta, p_c0, p_c1, p_c2, p_c3,
            );
            j += 1;
        }

        i += 4;
    }

    while i < m_end {
        let p_a = a.offset(i as isize * rsa);
        let p_c = c.offset(i as isize * rsc);

        let mut j = 0;
        while j + 8 <= n {
            let mut c_vec = _mm256_setzero_ps();
            for p in 0..k {
                let vb = _mm256_loadu_ps(b.offset(p as isize * rsb + j as isize));
                let va = _mm256_set1_ps(*p_a.offset(p as isize * csa));
                c_vec = _mm256_fmadd_ps(va, vb, c_vec);
            }
            if alpha != 1.0 {
                c_vec = _mm256_mul_ps(c_vec, _mm256_set1_ps(alpha));
            }
            store_c_vec(p_c.add(j), c_vec, beta);
            j += 8;
        }

        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *p_a.offset(p as isize * csa) * *b.offset(p as isize * rsb + j as isize);
            }
            let ptr_c = p_c.add(j);
            if beta == 0.0 {
                *ptr_c = alpha * sum;
            } else {
                *ptr_c = alpha * sum + beta * *ptr_c;
            }
            j += 1;
        }

        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store_c_tile(dst: *mut f32, c0: __m256, c1: __m256, beta: f32) {
    if beta == 0.0 {
        _mm256_storeu_ps(dst, c0);
        _mm256_storeu_ps(dst.add(8), c1);
    } else if beta == 1.0 {
        _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(dst), c0));
        _mm256_storeu_ps(dst.add(8), _mm256_add_ps(_mm256_loadu_ps(dst.add(8)), c1));
    } else {
        let v_beta = _mm256_set1_ps(beta);
        let orig0 = _mm256_mul_ps(_mm256_loadu_ps(dst), v_beta);
        let orig1 = _mm256_mul_ps(_mm256_loadu_ps(dst.add(8)), v_beta);
        _mm256_storeu_ps(dst, _mm256_add_ps(c0, orig0));
        _mm256_storeu_ps(dst.add(8), _mm256_add_ps(c1, orig1));
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store_c_vec(dst: *mut f32, c: __m256, beta: f32) {
    if beta == 0.0 {
        _mm256_storeu_ps(dst, c);
    } else if beta == 1.0 {
        _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(dst), c));
    } else {
        let v_beta = _mm256_set1_ps(beta);
        let orig = _mm256_mul_ps(_mm256_loadu_ps(dst), v_beta);
        _mm256_storeu_ps(dst, _mm256_add_ps(c, orig));
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn scalar_col_4(
    p_a0: *const f32,
    p_a1: *const f32,
    p_a2: *const f32,
    p_a3: *const f32,
    csa: isize,
    b: *const f32,
    rsb: isize,
    col_idx: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    p_c0: *mut f32,
    p_c1: *mut f32,
    p_c2: *mut f32,
    p_c3: *mut f32,
) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    for p in 0..k {
        let b_val = *b.offset(p as isize * rsb + col_idx as isize);
        sum0 += *p_a0.offset(p as isize * csa) * b_val;
        sum1 += *p_a1.offset(p as isize * csa) * b_val;
        sum2 += *p_a2.offset(p as isize * csa) * b_val;
        sum3 += *p_a3.offset(p as isize * csa) * b_val;
    }

    update_c(p_c0.add(col_idx), sum0, alpha, beta);
    update_c(p_c1.add(col_idx), sum1, alpha, beta);
    update_c(p_c2.add(col_idx), sum2, alpha, beta);
    update_c(p_c3.add(col_idx), sum3, alpha, beta);
}

#[inline(always)]
unsafe fn update_c(dst: *mut f32, sum: f32, alpha: f32, beta: f32) {
    if beta == 0.0 {
        *dst = alpha * sum;
    } else {
        *dst = alpha * sum + beta * *dst;
    }
}

/// Evaluates vectorized binary thresholding on 8-bit grayscale pixels.
///
/// # Safety
/// Pointers `src` and `dst` must point to valid allocations of length `n`.
pub unsafe fn binarize_u8(n: usize, src: *const u8, dst: *mut u8, threshold: u8, invert: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_supported() && n >= 32 {
            binarize_u8_avx2(n, src, dst, threshold, invert);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= 16 {
            binarize_u8_neon(n, src, dst, threshold, invert);
            return;
        }
    }

    let (high, low) = if invert { (0u8, 255u8) } else { (255u8, 0u8) };
    for i in 0..n {
        *dst.add(i) = if *src.add(i) > threshold { high } else { low };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn binarize_u8_avx2(n: usize, src: *const u8, dst: *mut u8, threshold: u8, invert: bool) {
    // Convert unsigned comparison (a > threshold) into signed comparison ((a ^ 0x80) > (threshold ^ 0x80))
    let sign_flip = _mm256_set1_epi8(-128i8);
    let v_thresh = _mm256_xor_si256(_mm256_set1_epi8(threshold as i8), sign_flip);
    let mut i = 0;

    while i + 32 <= n {
        let v_src = _mm256_loadu_si256(src.add(i) as *const __m256i);
        let v_signed = _mm256_xor_si256(v_src, sign_flip);
        let mask = _mm256_cmpgt_epi8(v_signed, v_thresh);
        let out = if invert {
            _mm256_andnot_si256(mask, _mm256_set1_epi8(-1i8))
        } else {
            mask
        };
        _mm256_storeu_si256(dst.add(i) as *mut __m256i, out);
        i += 32;
    }

    let (high, low) = if invert { (0u8, 255u8) } else { (255u8, 0u8) };
    while i < n {
        *dst.add(i) = if *src.add(i) > threshold { high } else { low };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binarize_u8_neon(n: usize, src: *const u8, dst: *mut u8, threshold: u8, invert: bool) {
    let v_thresh = vdupq_n_u8(threshold);
    let mut i = 0;

    while i + 16 <= n {
        let v_src = vld1q_u8(src.add(i));
        let mask = vcgtq_u8(v_src, v_thresh);
        let out = if invert { vmvnq_u8(mask) } else { mask };
        vst1q_u8(dst.add(i), out);
        i += 16;
    }

    let (high, low) = if invert { (0u8, 255u8) } else { (255u8, 0u8) };
    while i < n {
        *dst.add(i) = if *src.add(i) > threshold { high } else { low };
        i += 1;
    }
}

