//! High-performance cache-tiled FP32 General Matrix Multiplication (GEMM).
//!
//! Evaluates C = alpha * (A @ B) + beta * C with support for arbitrary row and column strides.
//! Accelerated via x86_64 AVX2 + FMA 4x16 register accumulation microkernels and Rayon work-stealing.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use rayon::prelude::*;

/// Computes C = alpha * A @ B + beta * C.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must point to valid allocations satisfying the specified dimensions and strides.
pub unsafe fn gemm_f32(
    m: usize,
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
    parallel: bool,
) {
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 || alpha == 0.0 {
        scale_c(m, n, beta, c, rsc, csc);
        return;
    }

    if !parallel || m < 32 || (m * n) < 16384 {
        gemm_serial_range(
            0, m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc,
        );
    } else {
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((m + num_threads - 1) / num_threads).max(8);
        let num_chunks = (m + chunk_size - 1) / chunk_size;

        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let m_start = chunk_idx * chunk_size;
            let m_end = (m_start + chunk_size).min(m);
            if m_start < m_end {
                let local_a = a_addr as *const f32;
                let local_b = b_addr as *const f32;
                let local_c = c_addr as *mut f32;
                unsafe {
                    gemm_serial_range(
                        m_start, m_end, n, k, alpha, local_a, rsa, csa, local_b, rsb, csb, beta,
                        local_c, rsc, csc,
                    );
                }
            }
        });
    }
}

unsafe fn scale_c(m: usize, n: usize, beta: f32, c: *mut f32, rsc: isize, csc: isize) {
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                *c.offset(i as isize * rsc + j as isize * csc) = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let ptr = c.offset(i as isize * rsc + j as isize * csc);
                *ptr *= beta;
            }
        }
    }
}

pub(crate) unsafe fn gemm_serial_range(
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
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && csb == 1
            && csc == 1
        {
            gemm_avx2_range(
                m_start, m_end, n, k, alpha, a, rsa, csa, b, rsb, beta, c, rsc,
            );
            return;
        }
    }

    gemm_fallback_range(
        m_start, m_end, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_avx2_range(
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
        let p_a0 = a.offset(i as isize * rsa);
        let p_a1 = a.offset((i + 1) as isize * rsa);
        let p_a2 = a.offset((i + 2) as isize * rsa);
        let p_a3 = a.offset((i + 3) as isize * rsa);

        let p_c0 = c.offset(i as isize * rsc);
        let p_c1 = c.offset((i + 1) as isize * rsc);
        let p_c2 = c.offset((i + 2) as isize * rsc);
        let p_c3 = c.offset((i + 3) as isize * rsc);

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
        _mm256_storeu_ps(dst, _mm256_fmadd_ps(_mm256_loadu_ps(dst), v_beta, c0));
        _mm256_storeu_ps(
            dst.add(8),
            _mm256_fmadd_ps(_mm256_loadu_ps(dst.add(8)), v_beta, c1),
        );
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
        _mm256_storeu_ps(dst, _mm256_fmadd_ps(_mm256_loadu_ps(dst), v_beta, c));
    }
}

#[inline(always)]
unsafe fn scalar_col_4(
    p_a0: *const f32,
    p_a1: *const f32,
    p_a2: *const f32,
    p_a3: *const f32,
    csa: isize,
    b: *const f32,
    rsb: isize,
    j: usize,
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
        let b_val = *b.offset(p as isize * rsb + j as isize);
        sum0 += *p_a0.offset(p as isize * csa) * b_val;
        sum1 += *p_a1.offset(p as isize * csa) * b_val;
        sum2 += *p_a2.offset(p as isize * csa) * b_val;
        sum3 += *p_a3.offset(p as isize * csa) * b_val;
    }

    apply_scalar_c(p_c0.add(j), sum0, alpha, beta);
    apply_scalar_c(p_c1.add(j), sum1, alpha, beta);
    apply_scalar_c(p_c2.add(j), sum2, alpha, beta);
    apply_scalar_c(p_c3.add(j), sum3, alpha, beta);
}

#[inline(always)]
unsafe fn apply_scalar_c(ptr: *mut f32, sum: f32, alpha: f32, beta: f32) {
    if beta == 0.0 {
        *ptr = alpha * sum;
    } else {
        *ptr = alpha * sum + beta * *ptr;
    }
}

unsafe fn gemm_fallback_range(
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
    for i in m_start..m_end {
        let row_a = a.offset(i as isize * rsa);
        let row_c = c.offset(i as isize * rsc);

        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *row_a.offset(p as isize * csa)
                    * *b.offset(p as isize * rsb + j as isize * csb);
            }

            let ptr_c = row_c.offset(j as isize * csc);
            if beta == 0.0 {
                *ptr_c = alpha * sum;
            } else {
                *ptr_c = alpha * sum + beta * *ptr_c;
            }
        }
    }
}
