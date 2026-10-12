//! 2D Block-Tiled Online Softmax Attention (FlashAttention for CPU).
//!
//! Evaluates Softmax(Q @ K.T * scale + mask) @ V in O(1) memory without materializing
//! the N x N attention matrix. Accelerated via AVX2 / FMA with Rayon work-stealing.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use rayon::prelude::*;

/// Computes FlashAttention on CPU with 2D block-tiling and online softmax.
///
/// # Safety
/// Pointers `q`, `k`, `v`, and `out` must point to valid memory buffers.
pub unsafe fn flash_attention_f32(
    batch_count: usize,
    sq: usize,
    sk: usize,
    d: usize,
    dv: usize,
    scale_factor: f32,
    is_gqa: bool,
    hq: usize,
    hkv: usize,
    gqa_group: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    mask: *const f32,
    out: *mut f32,
) {
    if batch_count == 0 || sq == 0 || sk == 0 || d == 0 || dv == 0 {
        return;
    }

    let q_batch_stride = sq * d;
    let k_batch_stride = sk * d;
    let v_batch_stride = sk * dv;
    let out_batch_stride = sq * dv;
    let mask_batch_stride = sq * sk;

    let total_rows = batch_count * sq;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if total_rows <= 16 {
                for row_idx in 0..total_rows {
                    let b = row_idx / sq;
                    let i = row_idx % sq;
                    compute_row_avx2(
                        b,
                        i,
                        sq,
                        sk,
                        d,
                        dv,
                        scale_factor,
                        is_gqa,
                        hq,
                        hkv,
                        gqa_group,
                        q_batch_stride,
                        k_batch_stride,
                        v_batch_stride,
                        out_batch_stride,
                        mask_batch_stride,
                        q,
                        k,
                        v,
                        mask,
                        out,
                    );
                }
            } else {
                let q_addr = q as usize;
                let k_addr = k as usize;
                let v_addr = v as usize;
                let mask_addr = mask as usize;
                let out_addr = out as usize;

                (0..total_rows)
                    .into_par_iter()
                    .with_min_len(8)
                    .for_each(|row_idx| {
                        let b = row_idx / sq;
                        let i = row_idx % sq;
                        let local_q = q_addr as *const f32;
                        let local_k = k_addr as *const f32;
                        let local_v = v_addr as *const f32;
                        let local_mask = mask_addr as *const f32;
                        let local_out = out_addr as *mut f32;

                        unsafe {
                            compute_row_avx2(
                                b,
                                i,
                                sq,
                                sk,
                                d,
                                dv,
                                scale_factor,
                                is_gqa,
                                hq,
                                hkv,
                                gqa_group,
                                q_batch_stride,
                                k_batch_stride,
                                v_batch_stride,
                                out_batch_stride,
                                mask_batch_stride,
                                local_q,
                                local_k,
                                local_v,
                                local_mask,
                                local_out,
                            );
                        }
                    });
            }
            return;
        }
    }

    if total_rows <= 16 {
        for row_idx in 0..total_rows {
            let b = row_idx / sq;
            let i = row_idx % sq;
            compute_row_scalar(
                b,
                i,
                sq,
                sk,
                d,
                dv,
                scale_factor,
                is_gqa,
                hq,
                hkv,
                gqa_group,
                q_batch_stride,
                k_batch_stride,
                v_batch_stride,
                out_batch_stride,
                mask_batch_stride,
                q,
                k,
                v,
                mask,
                out,
            );
        }
    } else {
        let q_addr = q as usize;
        let k_addr = k as usize;
        let v_addr = v as usize;
        let mask_addr = mask as usize;
        let out_addr = out as usize;

        (0..total_rows)
            .into_par_iter()
            .with_min_len(8)
            .for_each(|row_idx| {
                let b = row_idx / sq;
                let i = row_idx % sq;
                let local_q = q_addr as *const f32;
                let local_k = k_addr as *const f32;
                let local_v = v_addr as *const f32;
                let local_mask = mask_addr as *const f32;
                let local_out = out_addr as *mut f32;

                unsafe {
                    compute_row_scalar(
                        b,
                        i,
                        sq,
                        sk,
                        d,
                        dv,
                        scale_factor,
                        is_gqa,
                        hq,
                        hkv,
                        gqa_group,
                        q_batch_stride,
                        k_batch_stride,
                        v_batch_stride,
                        out_batch_stride,
                        mask_batch_stride,
                        local_q,
                        local_k,
                        local_v,
                        local_mask,
                        local_out,
                    );
                }
            });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_row_avx2(
    b: usize,
    i: usize,
    _sq: usize,
    sk: usize,
    d: usize,
    dv: usize,
    scale_factor: f32,
    is_gqa: bool,
    hq: usize,
    hkv: usize,
    gqa_group: usize,
    q_batch_stride: usize,
    k_batch_stride: usize,
    v_batch_stride: usize,
    out_batch_stride: usize,
    mask_batch_stride: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    mask: *const f32,
    out: *mut f32,
) {
    let b_kv = if is_gqa {
        let b_seq = b / hq;
        let cur_hq = b % hq;
        let cur_hkv = cur_hq / gqa_group;
        b_seq * hkv + cur_hkv
    } else {
        b
    };

    let p_q_i = q.add(b * q_batch_stride + i * d);
    let p_k_base = k.add(b_kv * k_batch_stride);
    let p_v_base = v.add(b_kv * v_batch_stride);
    let p_out_i = out.add(b * out_batch_stride + i * dv);
    let p_mask_row = if !mask.is_null() {
        mask.add(b * mask_batch_stride + i * sk)
    } else {
        core::ptr::null()
    };

    let mut stack_acc = [0.0f32; 512];
    let mut heap_acc: Option<std::vec::Vec<f32>> = None;
    let p_acc = if dv <= 512 {
        stack_acc.as_mut_ptr()
    } else {
        let mut vec = vec![0.0f32; dv];
        let ptr = vec.as_mut_ptr();
        heap_acc = Some(vec);
        ptr
    };

    for idx in 0..dv {
        *p_acc.add(idx) = 0.0;
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;

    const BC: usize = 32;
    let mut block_scores = [0.0f32; BC];
    let mut block_weights = [0.0f32; BC];

    let mut j0 = 0;
    while j0 < sk {
        let blen = BC.min(sk - j0);

        compute_block_dots_avx2(
            blen,
            j0,
            d,
            p_q_i,
            p_k_base,
            scale_factor,
            p_mask_row,
            &mut block_scores,
        );

        let mut m_block = f32::NEG_INFINITY;
        for idx in 0..blen {
            if block_scores[idx] > m_block {
                m_block = block_scores[idx];
            }
        }

        if m_block.is_infinite() && m_block < 0.0 {
            j0 += BC;
            continue;
        }

        let mut l_block = 0.0f32;
        for idx in 0..blen {
            let s = block_scores[idx];
            if s <= f32::NEG_INFINITY + 1000.0 {
                block_weights[idx] = 0.0;
            } else {
                let p = (s - m_block).exp();
                block_weights[idx] = p;
                l_block += p;
            }
        }

        let m_new = max_score.max(m_block);
        let alpha = if max_score.is_infinite() && max_score < 0.0 {
            0.0
        } else {
            (max_score - m_new).exp()
        };
        let beta = (m_block - m_new).exp();

        sum_exp = sum_exp * alpha + l_block * beta;
        max_score = m_new;

        if alpha != 1.0 {
            rescale_acc_avx2(p_acc, dv, alpha);
        }

        accumulate_v_avx2(p_acc, p_v_base, j0, blen, dv, &block_weights, beta);

        j0 += BC;
    }

    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
    normalize_out_avx2(p_acc, p_out_i, dv, inv_sum);

    let _ = heap_acc;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_block_dots_avx2(
    blen: usize,
    j0: usize,
    d: usize,
    p_q_i: *const f32,
    p_k_base: *const f32,
    scale_factor: f32,
    p_mask_row: *const f32,
    block_scores: &mut [f32; 32],
) {
    let mut k_idx = 0;
    while k_idx + 4 <= blen {
        let j = j0 + k_idx;
        let pk0 = p_k_base.add(j * d);
        let pk1 = pk0.add(d);
        let pk2 = pk1.add(d);
        let pk3 = pk2.add(d);

        let mut v_dot0 = _mm256_setzero_ps();
        let mut v_dot1 = _mm256_setzero_ps();
        let mut v_dot2 = _mm256_setzero_ps();
        let mut v_dot3 = _mm256_setzero_ps();

        let mut c = 0;
        while c + 8 <= d {
            let v_q = _mm256_loadu_ps(p_q_i.add(c));
            v_dot0 = _mm256_fmadd_ps(v_q, _mm256_loadu_ps(pk0.add(c)), v_dot0);
            v_dot1 = _mm256_fmadd_ps(v_q, _mm256_loadu_ps(pk1.add(c)), v_dot1);
            v_dot2 = _mm256_fmadd_ps(v_q, _mm256_loadu_ps(pk2.add(c)), v_dot2);
            v_dot3 = _mm256_fmadd_ps(v_q, _mm256_loadu_ps(pk3.add(c)), v_dot3);
            c += 8;
        }

        let mut dot0 = hsum_avx(v_dot0);
        let mut dot1 = hsum_avx(v_dot1);
        let mut dot2 = hsum_avx(v_dot2);
        let mut dot3 = hsum_avx(v_dot3);

        while c < d {
            let q_val = *p_q_i.add(c);
            dot0 += q_val * *pk0.add(c);
            dot1 += q_val * *pk1.add(c);
            dot2 += q_val * *pk2.add(c);
            dot3 += q_val * *pk3.add(c);
            c += 1;
        }

        block_scores[k_idx] = dot0 * scale_factor;
        block_scores[k_idx + 1] = dot1 * scale_factor;
        block_scores[k_idx + 2] = dot2 * scale_factor;
        block_scores[k_idx + 3] = dot3 * scale_factor;

        k_idx += 4;
    }

    while k_idx < blen {
        let j = j0 + k_idx;
        let pk = p_k_base.add(j * d);
        let mut v_dot = _mm256_setzero_ps();
        let mut c = 0;
        while c + 8 <= d {
            v_dot = _mm256_fmadd_ps(
                _mm256_loadu_ps(p_q_i.add(c)),
                _mm256_loadu_ps(pk.add(c)),
                v_dot,
            );
            c += 8;
        }
        let mut dot = hsum_avx(v_dot);
        while c < d {
            dot += *p_q_i.add(c) * *pk.add(c);
            c += 1;
        }
        block_scores[k_idx] = dot * scale_factor;
        k_idx += 1;
    }

    if !p_mask_row.is_null() {
        for idx in 0..blen {
            block_scores[idx] += *p_mask_row.add(j0 + idx);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let v128 = _mm_add_ps(vlow, vhigh);
    let shuf = _mm_movehl_ps(v128, v128);
    let sums = _mm_add_ps(v128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 1);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rescale_acc_avx2(p_acc: *mut f32, dv: usize, alpha: f32) {
    let v_alpha = _mm256_set1_ps(alpha);
    let mut cv = 0;
    while cv + 8 <= dv {
        let ptr = p_acc.add(cv);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), v_alpha));
        cv += 8;
    }
    while cv < dv {
        *p_acc.add(cv) *= alpha;
        cv += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_v_avx2(
    p_acc: *mut f32,
    p_v_base: *const f32,
    j0: usize,
    blen: usize,
    dv: usize,
    weights: &[f32; 32],
    beta: f32,
) {
    for k_idx in 0..blen {
        let w = weights[k_idx] * beta;
        if w == 0.0 {
            continue;
        }

        let pv = p_v_base.add((j0 + k_idx) * dv);
        let v_w = _mm256_set1_ps(w);
        let mut cv = 0;
        while cv + 16 <= dv {
            let ptr0 = p_acc.add(cv);
            let ptr1 = p_acc.add(cv + 8);
            let v_a0 = _mm256_loadu_ps(ptr0);
            let v_a1 = _mm256_loadu_ps(ptr1);
            let v_v0 = _mm256_loadu_ps(pv.add(cv));
            let v_v1 = _mm256_loadu_ps(pv.add(cv + 8));

            _mm256_storeu_ps(ptr0, _mm256_fmadd_ps(v_w, v_v0, v_a0));
            _mm256_storeu_ps(ptr1, _mm256_fmadd_ps(v_w, v_v1, v_a1));
            cv += 16;
        }
        while cv + 8 <= dv {
            let ptr = p_acc.add(cv);
            let v_a = _mm256_loadu_ps(ptr);
            let v_v = _mm256_loadu_ps(pv.add(cv));
            _mm256_storeu_ps(ptr, _mm256_fmadd_ps(v_w, v_v, v_a));
            cv += 8;
        }
        while cv < dv {
            *p_acc.add(cv) += w * *pv.add(cv);
            cv += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalize_out_avx2(p_acc: *const f32, p_out: *mut f32, dv: usize, inv_sum: f32) {
    let v_inv = _mm256_set1_ps(inv_sum);
    let mut cv = 0;
    while cv + 8 <= dv {
        let v_a = _mm256_loadu_ps(p_acc.add(cv));
        _mm256_storeu_ps(p_out.add(cv), _mm256_mul_ps(v_a, v_inv));
        cv += 8;
    }
    while cv < dv {
        *p_out.add(cv) = *p_acc.add(cv) * inv_sum;
        cv += 1;
    }
}

unsafe fn compute_row_scalar(
    b: usize,
    i: usize,
    _sq: usize,
    sk: usize,
    d: usize,
    dv: usize,
    scale_factor: f32,
    is_gqa: bool,
    hq: usize,
    hkv: usize,
    gqa_group: usize,
    q_batch_stride: usize,
    k_batch_stride: usize,
    v_batch_stride: usize,
    out_batch_stride: usize,
    mask_batch_stride: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    mask: *const f32,
    out: *mut f32,
) {
    let b_kv = if is_gqa {
        let b_seq = b / hq;
        let cur_hq = b % hq;
        let cur_hkv = cur_hq / gqa_group;
        b_seq * hkv + cur_hkv
    } else {
        b
    };

    let p_q_i = q.add(b * q_batch_stride + i * d);
    let p_k_base = k.add(b_kv * k_batch_stride);
    let p_v_base = v.add(b_kv * v_batch_stride);
    let p_out_i = out.add(b * out_batch_stride + i * dv);
    let p_mask_row = if !mask.is_null() {
        mask.add(b * mask_batch_stride + i * sk)
    } else {
        core::ptr::null()
    };

    let mut stack_acc = [0.0f32; 512];
    let mut heap_acc: Option<std::vec::Vec<f32>> = None;
    let p_acc = if dv <= 512 {
        stack_acc.as_mut_ptr()
    } else {
        let mut vec = vec![0.0f32; dv];
        let ptr = vec.as_mut_ptr();
        heap_acc = Some(vec);
        ptr
    };

    for idx in 0..dv {
        *p_acc.add(idx) = 0.0;
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;

    const BC: usize = 32;
    let mut block_scores = [0.0f32; BC];
    let mut block_weights = [0.0f32; BC];

    let mut j0 = 0;
    while j0 < sk {
        let blen = BC.min(sk - j0);

        for k_idx in 0..blen {
            let j = j0 + k_idx;
            let pk = p_k_base.add(j * d);
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += *p_q_i.add(c) * *pk.add(c);
            }
            let mut score = dot * scale_factor;
            if !p_mask_row.is_null() {
                score += *p_mask_row.add(j);
            }
            block_scores[k_idx] = score;
        }

        let mut m_block = f32::NEG_INFINITY;
        for idx in 0..blen {
            if block_scores[idx] > m_block {
                m_block = block_scores[idx];
            }
        }

        if m_block.is_infinite() && m_block < 0.0 {
            j0 += BC;
            continue;
        }

        let mut l_block = 0.0f32;
        for idx in 0..blen {
            let s = block_scores[idx];
            if s <= f32::NEG_INFINITY + 1000.0 {
                block_weights[idx] = 0.0;
            } else {
                let p = (s - m_block).exp();
                block_weights[idx] = p;
                l_block += p;
            }
        }

        let m_new = max_score.max(m_block);
        let alpha = if max_score.is_infinite() && max_score < 0.0 {
            0.0
        } else {
            (max_score - m_new).exp()
        };
        let beta = (m_block - m_new).exp();

        sum_exp = sum_exp * alpha + l_block * beta;
        max_score = m_new;

        if alpha != 1.0 {
            for cv in 0..dv {
                *p_acc.add(cv) *= alpha;
            }
        }

        for k_idx in 0..blen {
            let w = block_weights[k_idx] * beta;
            if w == 0.0 {
                continue;
            }
            let pv = p_v_base.add((j0 + k_idx) * dv);
            for cv in 0..dv {
                *p_acc.add(cv) += w * *pv.add(cv);
            }
        }

        j0 += BC;
    }

    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
    for cv in 0..dv {
        *p_out_i.add(cv) = *p_acc.add(cv) * inv_sum;
    }

    let _ = heap_acc;
}
