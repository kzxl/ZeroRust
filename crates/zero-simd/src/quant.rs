//! High-performance SIMD quantization microkernels (Q4_0, Q8_0, Q4_K, FP8).
//! Supports x86_64 (AVX2+FMA) and ARM64 (NEON).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

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

/// Evaluates dot product of a single row of Q4_0 quantized weights against an activation vector.
///
/// # Safety
/// Pointers must point to valid allocations.
pub unsafe fn compute_q4_0_row(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::microkernel::is_avx2_fma_supported() {
            compute_q4_0_row_avx2(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        compute_q4_0_row_neon(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
        return;
    }
    #[allow(unreachable_code)]
    compute_q4_0_row_scalar(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
}

/// Evaluates dot product of a single row of Q8_0 quantized weights against an activation vector.
///
/// # Safety
/// Pointers must point to valid allocations.
pub unsafe fn compute_q8_0_row(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::microkernel::is_avx2_fma_supported() {
            compute_q8_0_row_avx2(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        compute_q8_0_row_neon(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
        return;
    }
    #[allow(unreachable_code)]
    compute_q8_0_row_scalar(r, ptr_weight, p_act, p_dst, row_byte_stride, num_blocks);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn compute_q4_0_row_neon(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 18);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let vd = vdupq_n_f32(d);
        let p_qs = p_block.add(2);
        let p_a = p_act.add(b * 32);

        for j in (0..16).step_by(4) {
            let b0 = *p_qs.add(j);
            let b1 = *p_qs.add(j + 1);
            let b2 = *p_qs.add(j + 2);
            let b3 = *p_qs.add(j + 3);

            let lo = [
                ((b0 & 0x0F) as i32 - 8) as f32,
                ((b1 & 0x0F) as i32 - 8) as f32,
                ((b2 & 0x0F) as i32 - 8) as f32,
                ((b3 & 0x0F) as i32 - 8) as f32,
            ];
            let hi = [
                ((b0 >> 4) as i32 - 8) as f32,
                ((b1 >> 4) as i32 - 8) as f32,
                ((b2 >> 4) as i32 - 8) as f32,
                ((b3 >> 4) as i32 - 8) as f32,
            ];

            let vlo = vmulq_f32(vld1q_f32(lo.as_ptr()), vd);
            let vhi = vmulq_f32(vld1q_f32(hi.as_ptr()), vd);

            let va_lo = vld1q_f32(p_a.add(j));
            let va_hi = vld1q_f32(p_a.add(j + 16));

            acc0 = vfmaq_f32(acc0, vlo, va_lo);
            acc1 = vfmaq_f32(acc1, vhi, va_hi);
        }
    }

    *p_dst.add(r) = vaddvq_f32(vaddq_f32(acc0, acc1));
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn compute_q8_0_row_neon(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 34);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let vd = vdupq_n_f32(d);
        let p_qs = p_block.add(2) as *const i8;
        let p_a = p_act.add(b * 32);

        for j in (0..32).step_by(8) {
            let q8 = vld1_s8(p_qs.add(j));
            let q16 = vmovl_s8(q8);
            let q32_lo = vmovl_s16(vget_low_s16(q16));
            let q32_hi = vmovl_s16(vget_high_s16(q16));

            let qf_lo = vmulq_f32(vcvtq_f32_s32(q32_lo), vd);
            let qf_hi = vmulq_f32(vcvtq_f32_s32(q32_hi), vd);

            let va_lo = vld1q_f32(p_a.add(j));
            let va_hi = vld1q_f32(p_a.add(j + 4));

            acc0 = vfmaq_f32(acc0, qf_lo, va_lo);
            acc1 = vfmaq_f32(acc1, qf_hi, va_hi);
        }
    }

    *p_dst.add(r) = vaddvq_f32(vaddq_f32(acc0, acc1));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_q4_0_row_avx2(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let low_mask = _mm256_set1_epi8(0x0F);
    let offset_8 = _mm256_set1_epi16(8);

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 18);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let vd = _mm256_set1_ps(d);

        let p_qs = p_block.add(2);
        let raw_bytes = _mm_loadu_si128(p_qs as *const __m128i);
        let raw_256 = _mm256_castsi128_si256(raw_bytes);
        let raw_256 = _mm256_insertf128_si256(raw_256, raw_bytes, 1);

        let low_nibbles = _mm256_and_si256(raw_256, low_mask);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(raw_256, 4), low_mask);

        let low16 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(low_nibbles)), offset_8);
        let high16 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(high_nibbles)), offset_8);

        let q0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(low16));
        let q1_i32 = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(low16, 1));
        let q2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(high16));
        let q3_i32 = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(high16, 1));

        let q0_f = _mm256_cvtepi32_ps(q0_i32);
        let q1_f = _mm256_cvtepi32_ps(q1_i32);
        let q2_f = _mm256_cvtepi32_ps(q2_i32);
        let q3_f = _mm256_cvtepi32_ps(q3_i32);

        let p_act_block = p_act.add(b * 32);
        let act0 = _mm256_loadu_ps(p_act_block);
        let act1 = _mm256_loadu_ps(p_act_block.add(8));
        let act2 = _mm256_loadu_ps(p_act_block.add(16));
        let act3 = _mm256_loadu_ps(p_act_block.add(24));

        let dot0 = _mm256_fmadd_ps(q0_f, act0, _mm256_mul_ps(q1_f, act1));
        let dot1 = _mm256_fmadd_ps(q2_f, act2, _mm256_mul_ps(q3_f, act3));

        acc0 = _mm256_fmadd_ps(vd, dot0, acc0);
        acc1 = _mm256_fmadd_ps(vd, dot1, acc1);
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    *p_dst.add(r) = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_q8_0_row_avx2(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 34);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let vd = _mm256_set1_ps(d);

        let p_qs = p_block.add(2);
        let raw16_0 = _mm_loadu_si128(p_qs as *const __m128i);
        let raw16_1 = _mm_loadu_si128(p_qs.add(16) as *const __m128i);

        let q0_i16 = _mm256_cvtepi8_epi16(raw16_0);
        let q1_i16 = _mm256_cvtepi8_epi16(raw16_1);

        let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q0_i16)));
        let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(q0_i16, 1)));
        let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q1_i16)));
        let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(q1_i16, 1)));

        let p_act_block = p_act.add(b * 32);
        let act0 = _mm256_loadu_ps(p_act_block);
        let act1 = _mm256_loadu_ps(p_act_block.add(8));
        let act2 = _mm256_loadu_ps(p_act_block.add(16));
        let act3 = _mm256_loadu_ps(p_act_block.add(24));

        let dot0 = _mm256_fmadd_ps(q0_f, act0, _mm256_mul_ps(q1_f, act1));
        let dot1 = _mm256_fmadd_ps(q2_f, act2, _mm256_mul_ps(q3_f, act3));

        acc0 = _mm256_fmadd_ps(vd, dot0, acc0);
        acc1 = _mm256_fmadd_ps(vd, dot1, acc1);
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    *p_dst.add(r) = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

pub unsafe fn compute_q4_0_row_scalar(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 18);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let p_qs = p_block.add(2);
        let p_act_block = p_act.add(b * 32);

        for j in 0..16 {
            let byte = *p_qs.add(j);
            let q0 = ((byte & 0x0F) as i8 - 8) as f32;
            let q1 = (((byte >> 4) & 0x0F) as i8 - 8) as f32;

            sum += d * (q0 * *p_act_block.add(j) + q1 * *p_act_block.add(j + 16));
        }
    }

    *p_dst.add(r) = sum;
}

pub unsafe fn compute_q8_0_row_scalar(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let p_block = p_row.add(b * 34);
        let raw_d = (p_block as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let p_qs = p_block.add(2) as *const i8;
        let p_act_block = p_act.add(b * 32);

        for j in 0..32 {
            let q = *p_qs.add(j) as f32;
            sum += d * (q * *p_act_block.add(j));
        }
    }

    *p_dst.add(r) = sum;
}

pub const QK_K: usize = 256;
pub const BYTES_PER_BLOCK_Q4_K: usize = 144;

/// Converts 8-bit floating point FP8 (E4M3 format) to IEEE 754 32-bit float.
#[inline(always)]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = ((byte >> 7) & 1) as u32;
    let exp = ((byte >> 3) & 0x0F) as u32;
    let mant = (byte & 0x07) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let val = (mant as f32) / 8.0f32 * (1.0f32 / 64.0f32);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        let new_exp = exp + (127 - 7);
        let bits = (sign << 31) | (new_exp << 23) | (mant << 20);
        f32::from_bits(bits)
    }
}

/// Evaluates dot product of a single row of FP8 (E4M3) weights against an activation vector.
///
/// # Safety
/// Pointers must point to valid allocations.
pub unsafe fn compute_fp8_row(
    r: usize,
    ptr_weight: *const u8,
    scale: f32,
    p_act: *const f32,
    p_dst: *mut f32,
    k_cols: usize,
) {
    let p_row = ptr_weight.add(r * k_cols);
    let mut sum = 0.0f32;
    for i in 0..k_cols {
        let w = fp8_e4m3_to_f32(*p_row.add(i)) * scale;
        sum += w * *p_act.add(i);
    }
    *p_dst.add(r) = sum;
}

/// Evaluates dot product of a single row of Q4_K quantized weights against an activation vector.
///
/// # Safety
/// Pointers must point to valid allocations.
pub unsafe fn compute_q4_k_row(
    r: usize,
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    row_byte_stride: usize,
    num_super_blocks: usize,
) {
    let p_row = ptr_weight.add(r * row_byte_stride);
    let mut sum = 0.0f32;

    for sb in 0..num_super_blocks {
        let p_sb = p_row.add(sb * BYTES_PER_BLOCK_Q4_K);
        let raw_d = (p_sb as *const u16).read_unaligned();
        let raw_dmin = (p_sb.add(2) as *const u16).read_unaligned();
        let d = f16_to_f32(raw_d);
        let dmin = f16_to_f32(raw_dmin);

        let p_scales = p_sb.add(4);
        let p_qs = p_sb.add(16);
        let p_act_sb = p_act.add(sb * QK_K);

        // 8 sub-blocks of 32 weights each
        for sub in 0..8 {
            let sc = *p_scales.add(sub) as f32;
            let m = *p_scales.add(sub + 4) as f32;
            let d_sub = d * sc;
            let m_sub = dmin * m;

            let p_act_sub = p_act_sb.add(sub * 32);
            let qs_offset = sub * 16;

            for j in 0..16 {
                let byte = *p_qs.add(qs_offset + j);
                let q0 = (byte & 0x0F) as f32;
                let q1 = ((byte >> 4) & 0x0F) as f32;

                let w0 = d_sub * q0 - m_sub;
                let w1 = d_sub * q1 - m_sub;

                sum += w0 * *p_act_sub.add(j) + w1 * *p_act_sub.add(j + 16);
            }
        }
    }

    *p_dst.add(r) = sum;
}
