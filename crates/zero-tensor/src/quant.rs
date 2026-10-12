//! Streaming on-the-fly Quantized Matrix-Vector multiplication (GEMV).
//!
//! Evaluates y = x @ W^T directly on memory-mapped raw quantized byte streams
//! without materializing or dequantizing the full weight matrix into DRAM.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use rayon::prelude::*;

const QK4_0: usize = 32;
const QK8_0: usize = 32;

/// Converts IEEE 754 16-bit half-precision float to 32-bit single-precision float.
#[inline(always)]
pub fn f16_to_f32(raw: u16) -> f32 {
    let sign = ((raw >> 15) & 1) as u32;
    let exp = ((raw >> 10) & 0x1F) as u32;
    let mant = (raw & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 113u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits((sign << 31) | (e << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        let new_exp = exp + 112;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

/// Streaming on-the-fly Q4_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers must point to valid allocations satisfying the specified dimensions.
pub unsafe fn gemv_q4_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) {
    if k_cols % QK4_0 != 0 || n_rows == 0 {
        return;
    }

    let num_blocks_per_row = k_cols / QK4_0;
    const BYTES_PER_BLOCK: usize = 18;
    let row_byte_stride = num_blocks_per_row * BYTES_PER_BLOCK;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if n_rows < 32 {
                for r in 0..n_rows {
                    compute_q4_0_row_avx2(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks_per_row);
                }
            } else {
                let weight_addr = ptr_weight as usize;
                let act_addr = p_act as usize;
                let dst_addr = p_dst as usize;

                (0..n_rows).into_par_iter().with_min_len(64).for_each(|r| {
                    let local_w = weight_addr as *const u8;
                    let local_act = act_addr as *const f32;
                    let local_dst = dst_addr as *mut f32;
                    unsafe {
                        compute_q4_0_row_avx2(r, local_w, local_act, local_dst, row_byte_stride, num_blocks_per_row);
                    }
                });
            }
            return;
        }
    }

    if n_rows < 32 {
        for r in 0..n_rows {
            compute_q4_0_row_scalar(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks_per_row);
        }
    } else {
        let weight_addr = ptr_weight as usize;
        let act_addr = p_act as usize;
        let dst_addr = p_dst as usize;

        (0..n_rows).into_par_iter().with_min_len(64).for_each(|r| {
            let local_w = weight_addr as *const u8;
            let local_act = act_addr as *const f32;
            let local_dst = dst_addr as *mut f32;
            unsafe {
                compute_q4_0_row_scalar(r, local_w, local_act, local_dst, row_byte_stride, num_blocks_per_row);
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_q4_0_row_avx2(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks_per_row: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;
    let mut act_offset = 0;

    let v_mask = _mm256_set1_epi32(0x0F);
    let v_offset = _mm256_set1_epi32(8);

    for b in 0..num_blocks_per_row {
        let block_ptr = p_row.add(b * 18);
        let raw_scale = core::ptr::read_unaligned(block_ptr as *const u16);
        let d = f16_to_f32(raw_scale);
        let qs = block_ptr.add(2);

        let raw0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs as *const _));
        let raw1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs.add(8) as *const _));

        let w0_0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(raw0, v_mask), v_offset));
        let w1_0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
            _mm256_and_si256(_mm256_srli_epi32(raw0, 4), v_mask),
            v_offset,
        ));

        let w0_1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(raw1, v_mask), v_offset));
        let w1_1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
            _mm256_and_si256(_mm256_srli_epi32(raw1, 4), v_mask),
            v_offset,
        ));

        let v_dot0 = _mm256_mul_ps(w0_0, _mm256_loadu_ps(p_act.add(act_offset)));
        let v_dot1 = _mm256_mul_ps(w0_1, _mm256_loadu_ps(p_act.add(act_offset + 8)));
        let v_dot2 = _mm256_fmadd_ps(w1_0, _mm256_loadu_ps(p_act.add(act_offset + 16)), v_dot0);
        let v_dot3 = _mm256_fmadd_ps(w1_1, _mm256_loadu_ps(p_act.add(act_offset + 24)), v_dot1);

        sum += hsum_avx(_mm256_add_ps(v_dot2, v_dot3)) * d;
        act_offset += QK4_0;
    }

    *p_dst.add(r) = sum;
}

unsafe fn compute_q4_0_row_scalar(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks_per_row: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;
    let mut act_offset = 0;

    for b in 0..num_blocks_per_row {
        let block_ptr = p_row.add(b * 18);
        let raw_scale = core::ptr::read_unaligned(block_ptr as *const u16);
        let d = f16_to_f32(raw_scale);
        let qs = block_ptr.add(2);

        let mut block_sum = 0.0f32;
        for j in 0..16 {
            let val = *qs.add(j);
            let x0 = ((val & 0x0F) as i32) - 8;
            let x1 = (((val >> 4) & 0x0F) as i32) - 8;
            block_sum += *p_act.add(act_offset + j) * (x0 as f32)
                + *p_act.add(act_offset + j + 16) * (x1 as f32);
        }

        sum += block_sum * d;
        act_offset += QK4_0;
    }

    *p_dst.add(r) = sum;
}

/// Streaming on-the-fly Q8_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers must point to valid allocations satisfying the specified dimensions.
pub unsafe fn gemv_q8_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) {
    if k_cols % QK8_0 != 0 || n_rows == 0 {
        return;
    }

    let num_blocks_per_row = k_cols / QK8_0;
    const BYTES_PER_BLOCK: usize = 34;
    let row_byte_stride = num_blocks_per_row * BYTES_PER_BLOCK;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if n_rows < 32 {
                for r in 0..n_rows {
                    compute_q8_0_row_avx2(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks_per_row);
                }
            } else {
                let weight_addr = ptr_weight as usize;
                let act_addr = p_act as usize;
                let dst_addr = p_dst as usize;

                (0..n_rows).into_par_iter().with_min_len(64).for_each(|r| {
                    let local_w = weight_addr as *const u8;
                    let local_act = act_addr as *const f32;
                    let local_dst = dst_addr as *mut f32;
                    unsafe {
                        compute_q8_0_row_avx2(r, local_w, local_act, local_dst, row_byte_stride, num_blocks_per_row);
                    }
                });
            }
            return;
        }
    }

    if n_rows < 32 {
        for r in 0..n_rows {
            compute_q8_0_row_scalar(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks_per_row);
        }
    } else {
        let weight_addr = ptr_weight as usize;
        let act_addr = p_act as usize;
        let dst_addr = p_dst as usize;

        (0..n_rows).into_par_iter().with_min_len(64).for_each(|r| {
            let local_w = weight_addr as *const u8;
            let local_act = act_addr as *const f32;
            let local_dst = dst_addr as *mut f32;
            unsafe {
                compute_q8_0_row_scalar(r, local_w, local_act, local_dst, row_byte_stride, num_blocks_per_row);
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_q8_0_row_avx2(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks_per_row: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;
    let mut act_offset = 0;

    for b in 0..num_blocks_per_row {
        let block_ptr = p_row.add(b * 34);
        let raw_scale = core::ptr::read_unaligned(block_ptr as *const u16);
        let d = f16_to_f32(raw_scale);
        let qs = block_ptr.add(2) as *const i8;

        let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs as *const _)));
        let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(8) as *const _)));
        let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(16) as *const _)));
        let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(24) as *const _)));

        let v_dot0 = _mm256_mul_ps(w0, _mm256_loadu_ps(p_act.add(act_offset)));
        let v_dot1 = _mm256_mul_ps(w1, _mm256_loadu_ps(p_act.add(act_offset + 8)));
        let v_dot2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p_act.add(act_offset + 16)), v_dot0);
        let v_dot3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p_act.add(act_offset + 24)), v_dot1);

        sum += hsum_avx(_mm256_add_ps(v_dot2, v_dot3)) * d;
        act_offset += QK8_0;
    }

    *p_dst.add(r) = sum;
}

unsafe fn compute_q8_0_row_scalar(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks_per_row: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;
    let mut act_offset = 0;

    for b in 0..num_blocks_per_row {
        let block_ptr = p_row.add(b * 34);
        let raw_scale = core::ptr::read_unaligned(block_ptr as *const u16);
        let d = f16_to_f32(raw_scale);
        let qs = block_ptr.add(2) as *const i8;

        let mut block_sum = 0.0f32;
        for j in 0..32 {
            block_sum += *p_act.add(act_offset + j) * (*qs.add(j) as f32);
        }

        sum += block_sum * d;
        act_offset += QK8_0;
    }

    *p_dst.add(r) = sum;
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
