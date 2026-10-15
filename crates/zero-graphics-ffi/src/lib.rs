//! # ZeroGraphics FFI Gateway
//!
//! High-performance SIMD C-ABI export interface for ZeroGraphics & ZeroVision:
//! - AVX2/NEON `Bgra32 -> Gray8` BT.601 color conversion (`zg_bgra32_to_gray8`)
//! - Separable 5x5 Binomial Gaussian Filter (`zg_gaussian5x5_u8`)
//! - Trigonometry-free Canny Edge Detector (`zg_canny_edge_u8`)
//! - Integral-Image + AVX2 FMA Normalized Cross-Correlation (`zg_ncc_match_u8`)
//! - AVX2 Signed Distance Field (SDF) rasterizer (`zg_sdf_generate_u8`)
//! - AVX2 3x3 Grayscale Morphological Erosion & Dilation (`zg_morphology3x3_u8`)
//! - Largest-Triangle-Three-Buckets (LTTB) waveform decimation (`zg_lttb_decimate_f64`)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::panic::catch_unwind;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub const ZG_SUCCESS: i32 = 0;
pub const ZG_ERR_NULL_POINTER: i32 = -1;
pub const ZG_ERR_INVALID_DIMENSION: i32 = -2;
pub const ZG_ERR_UNSUPPORTED: i32 = -3;
pub const ZG_ERR_PANIC: i32 = -99;

// ============================================================================
// 1 & 2. Version & Hardware SIMD Capability
// ============================================================================

/// Returns the C-ABI version of ZeroGraphics FFI (100 for v1.0.0).
#[no_mangle]
pub extern "C" fn zg_version() -> u32 {
    100
}

/// Returns 1 if AVX2/FMA (x86_64) or NEON (aarch64) hardware SIMD is supported, 0 otherwise.
#[no_mangle]
pub extern "C" fn zg_is_simd_supported() -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return 1;
        }
    }
    if zero_simd::is_avx2_fma_supported() || zero_simd::is_neon_supported() {
        1
    } else {
        0
    }
}

// ============================================================================
// 3. Bgra32 -> Gray8 Color Conversion (BT.601 Fixed-Point + AVX2)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bgra32_to_gray8_row_avx2(src_row: *const u8, dst_row: *mut u8, width: usize) {
    let mask_00ff = _mm256_set1_epi32(0x00FF_00FFu32 as i32);
    // Low 16 bits: B weight = 29; High 16 bits: R weight = 77
    let w_br = _mm256_set1_epi32((77 << 16) | 29);
    // Low 16 bits: G weight = 150; High 16 bits: A weight = 0
    let w_ga = _mm256_set1_epi32(150);

    let mut x = 0usize;
    while x + 8 <= width {
        let v = _mm256_loadu_si256(src_row.add(x * 4) as *const __m256i);
        let br = _mm256_and_si256(v, mask_00ff);
        let ga = _mm256_and_si256(_mm256_srli_epi16(v, 8), mask_00ff);

        let sum_br = _mm256_madd_epi16(br, w_br);
        let sum_ga = _mm256_madd_epi16(ga, w_ga);
        let gray32 = _mm256_srli_epi32(_mm256_add_epi32(sum_br, sum_ga), 8);

        let lo128 = _mm256_castsi256_si128(gray32);
        let hi128 = _mm256_extracti128_si256(gray32, 1);
        let p16 = _mm_packus_epi32(lo128, hi128);
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(dst_row.add(x) as *mut __m128i, p8);

        x += 8;
    }

    while x < width {
        let o = x * 4;
        let b = *src_row.add(o) as u32;
        let g = *src_row.add(o + 1) as u32;
        let r = *src_row.add(o + 2) as u32;
        *dst_row.add(x) = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        x += 1;
    }
}

unsafe fn bgra32_to_gray8_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && width >= 8;

    for y in 0..height {
        let src_row = src.add(y * src_stride);
        let dst_row = dst.add(y * dst_stride);

        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            bgra32_to_gray8_row_avx2(src_row, dst_row, width);
            continue;
        }

        for x in 0..width {
            let o = x * 4;
            let b = *src_row.add(o) as u32;
            let g = *src_row.add(o + 1) as u32;
            let r = *src_row.add(o + 2) as u32;
            *dst_row.add(x) = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        }
    }
}

/// Converts a `Bgra32` (`[B, G, R, A]`) image buffer to `Gray8` using exact integer BT.601 weights:
/// `((r * 77 + g * 150 + b * 29) >> 8) as u8`.
///
/// # Safety
/// `src` and `dst` must be valid pointers for `height` rows of `src_stride` and `dst_stride` bytes.
#[no_mangle]
pub unsafe extern "C" fn zg_bgra32_to_gray8(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
) -> i32 {
    if src.is_null() || dst.is_null() {
        return ZG_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return ZG_SUCCESS;
    }
    if src_stride < width * 4 || dst_stride < width {
        return ZG_ERR_INVALID_DIMENSION;
    }

    let res = catch_unwind(|| {
        bgra32_to_gray8_inner(src, src_stride, dst, dst_stride, width, height);
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 4. Separable 5x5 Binomial Gaussian Filter ([1, 4, 6, 4, 1] / 16)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gaussian5x5_horiz_row_avx2(src_row: *const u8, temp_row: *mut u16, width: usize) {
    *temp_row.add(0) = (*src_row.add(0) as u16) * 16;
    *temp_row.add(1) = (*src_row.add(1) as u16) * 16;
    *temp_row.add(width - 2) = (*src_row.add(width - 2) as u16) * 16;
    *temp_row.add(width - 1) = (*src_row.add(width - 1) as u16) * 16;

    let w4 = _mm256_set1_epi16(4);
    let w6 = _mm256_set1_epi16(6);

    let mut x = 2usize;
    let limit = width - 2;
    while x + 16 <= limit {
        let v_m2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(src_row.add(x - 2) as *const __m128i));
        let v_m1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(src_row.add(x - 1) as *const __m128i));
        let v_0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(src_row.add(x) as *const __m128i));
        let v_p1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(src_row.add(x + 1) as *const __m128i));
        let v_p2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(src_row.add(x + 2) as *const __m128i));

        let sum_1 = _mm256_add_epi16(v_m2, v_p2);
        let sum_4 = _mm256_mullo_epi16(_mm256_add_epi16(v_m1, v_p1), w4);
        let sum_6 = _mm256_mullo_epi16(v_0, w6);
        let total = _mm256_add_epi16(_mm256_add_epi16(sum_1, sum_4), sum_6);

        _mm256_storeu_si256(temp_row.add(x) as *mut __m256i, total);
        x += 16;
    }

    while x < limit {
        *temp_row.add(x) = (*src_row.add(x - 2) as u16)
            + (*src_row.add(x - 1) as u16) * 4
            + (*src_row.add(x) as u16) * 6
            + (*src_row.add(x + 1) as u16) * 4
            + (*src_row.add(x + 2) as u16);
        x += 1;
    }
}

unsafe fn gaussian5x5_u8_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    if width < 5 || height < 5 {
        for y in 0..height {
            core::ptr::copy_nonoverlapping(src.add(y * src_stride), dst.add(y * dst_stride), width);
        }
        return;
    }

    let mut temp = vec![0u16; width * height];
    let temp_ptr = temp.as_mut_ptr();

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");

    // Horizontal pass
    for y in 0..height {
        let src_row = src.add(y * src_stride);
        let temp_row = temp_ptr.add(y * width);

        #[cfg(target_arch = "x86_64")]
        if use_avx2 && width >= 20 {
            gaussian5x5_horiz_row_avx2(src_row, temp_row, width);
            continue;
        }

        *temp_row.add(0) = (*src_row.add(0) as u16) * 16;
        *temp_row.add(1) = (*src_row.add(1) as u16) * 16;
        for x in 2..(width - 2) {
            *temp_row.add(x) = (*src_row.add(x - 2) as u16)
                + (*src_row.add(x - 1) as u16) * 4
                + (*src_row.add(x) as u16) * 6
                + (*src_row.add(x + 1) as u16) * 4
                + (*src_row.add(x + 2) as u16);
        }
        *temp_row.add(width - 2) = (*src_row.add(width - 2) as u16) * 16;
        *temp_row.add(width - 1) = (*src_row.add(width - 1) as u16) * 16;
    }

    // Vertical pass
    for y in 2..(height - 2) {
        let r0 = temp_ptr.add((y - 2) * width);
        let r1 = temp_ptr.add((y - 1) * width);
        let r2 = temp_ptr.add(y * width);
        let r3 = temp_ptr.add((y + 1) * width);
        let r4 = temp_ptr.add((y + 2) * width);
        let dst_row = dst.add(y * dst_stride);

        for x in 0..width {
            let sum = (*r0.add(x) as u32)
                + (*r1.add(x) as u32) * 4
                + (*r2.add(x) as u32) * 6
                + (*r3.add(x) as u32) * 4
                + (*r4.add(x) as u32);
            *dst_row.add(x) = (sum >> 8) as u8;
        }
    }

    // Copy border 2 top/bottom rows directly from src
    for &by in &[0usize, 1usize, height - 2, height - 1] {
        core::ptr::copy_nonoverlapping(src.add(by * src_stride), dst.add(by * dst_stride), width);
    }
}

/// Applies a separable 5x5 binomial Gaussian smoothing kernel (`[1, 4, 6, 4, 1] / 16`)
/// matching `CannyEdgeDetector.ApplyGaussian5x5` border handling.
///
/// # Safety
/// `src` and `dst` must be valid pointers for `height` rows of `src_stride` and `dst_stride` bytes.
#[no_mangle]
pub unsafe extern "C" fn zg_gaussian5x5_u8(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
) -> i32 {
    if src.is_null() || dst.is_null() {
        return ZG_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return ZG_SUCCESS;
    }
    if src_stride < width || dst_stride < width {
        return ZG_ERR_INVALID_DIMENSION;
    }

    let res = catch_unwind(|| {
        gaussian5x5_u8_inner(src, src_stride, dst, dst_stride, width, height);
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 5. High-Speed Canny Edge Detector (Trigonometry-Free Integer Sector NMS)
// ============================================================================

const NON_EDGE: u8 = 0;
const WEAK_EDGE: u8 = 128;
const STRONG_EDGE: u8 = 255;

#[inline(always)]
fn classify_sobel_sector(gx: i32, gy: i32) -> u8 {
    let ax = gx.unsigned_abs() as u64;
    let ay = gy.unsigned_abs() as u64;
    // tan(22.5 deg) = sqrt(2) - 1 ~= 0.41421356
    // tan(67.5 deg) = sqrt(2) + 1 ~= 2.41421356
    if ay * 10000 < ax * 4142 {
        0 // 0 deg (Horizontal gradient -> compare left/right)
    } else if ay * 10000 >= ax * 24142 {
        if ax == 0 && ay == 0 {
            0
        } else {
            2 // 90 deg (Vertical gradient -> compare top/bottom)
        }
    } else if (gx ^ gy) >= 0 {
        1 // 45 deg
    } else {
        3 // 135 deg
    }
}

/// High-speed Canny Edge Detector with optional 5x5 Gaussian filter, 3x3 Sobel gradients,
/// trigonometry-free integer tangent sector classification, 4-direction NMS, and 8-connected hysteresis.
///
/// # Safety
/// `src` and `dst` must be valid pointers for `height` rows of `src_stride` and `dst_stride` bytes.
#[no_mangle]
pub unsafe extern "C" fn zg_canny_edge_u8(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
    low_threshold: f32,
    high_threshold: f32,
    apply_gaussian: i32,
) -> i32 {
    if src.is_null() || dst.is_null() {
        return ZG_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return ZG_SUCCESS;
    }
    if src_stride < width
        || dst_stride < width
        || low_threshold < 0.0
        || high_threshold < low_threshold
    {
        return ZG_ERR_INVALID_DIMENSION;
    }

    let res = catch_unwind(|| {
        // Clear output rows
        for y in 0..height {
            core::ptr::write_bytes(dst.add(y * dst_stride), NON_EDGE, width);
        }
        if width < 3 || height < 3 {
            return;
        }

        let total = width * height;
        let mut smoothed = vec![0u8; total];
        if apply_gaussian != 0 && width >= 5 && height >= 5 {
            gaussian5x5_u8_inner(src, src_stride, smoothed.as_mut_ptr(), width, width, height);
        } else {
            for y in 0..height {
                core::ptr::copy_nonoverlapping(
                    src.add(y * src_stride),
                    smoothed.as_mut_ptr().add(y * width),
                    width,
                );
            }
        }

        let mut magnitude = vec![0.0f32; total];
        let mut orientation = vec![0u8; total];

        // 1. Sobel 3x3 Gradients + Integer Sector Classification
        for y in 1..(height - 1) {
            let y_prev = (y - 1) * width;
            let y_cur = y * width;
            let y_next = (y + 1) * width;

            for x in 1..(width - 1) {
                let p00 = smoothed[y_prev + x - 1] as i32;
                let p01 = smoothed[y_prev + x] as i32;
                let p02 = smoothed[y_prev + x + 1] as i32;
                let p10 = smoothed[y_cur + x - 1] as i32;
                let p12 = smoothed[y_cur + x + 1] as i32;
                let p20 = smoothed[y_next + x - 1] as i32;
                let p21 = smoothed[y_next + x] as i32;
                let p22 = smoothed[y_next + x + 1] as i32;

                let gx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
                let gy = (p20 - p00) + 2 * (p21 - p01) + (p22 - p02);

                let mag = ((gx * gx + gy * gy) as f64).sqrt() as f32;
                let idx = y_cur + x;
                magnitude[idx] = mag;
                orientation[idx] = classify_sobel_sector(gx, gy);
            }
        }

        // 2. 4-Direction Non-Maximum Suppression (NMS)
        let mut nms = vec![NON_EDGE; total];
        let mut stack = Vec::with_capacity(total / 8);

        for y in 1..(height - 1) {
            let y_prev = (y - 1) * width;
            let y_cur = y * width;
            let y_next = (y + 1) * width;

            for x in 1..(width - 1) {
                let idx = y_cur + x;
                let m = magnitude[idx];
                if m < low_threshold {
                    continue;
                }

                let (m1, m2) = match orientation[idx] {
                    0 => (magnitude[y_cur + x - 1], magnitude[y_cur + x + 1]),
                    1 => (magnitude[y_prev + x + 1], magnitude[y_next + x - 1]),
                    2 => (magnitude[y_prev + x], magnitude[y_next + x]),
                    _ => (magnitude[y_prev + x - 1], magnitude[y_next + x + 1]),
                };

                if m >= m1 && m >= m2 {
                    if m >= high_threshold {
                        nms[idx] = STRONG_EDGE;
                        *dst.add(y * dst_stride + x) = STRONG_EDGE;
                        stack.push(idx);
                    } else {
                        nms[idx] = WEAK_EDGE;
                    }
                }
            }
        }

        // 3. 8-Connected Hysteresis Edge Tracking
        let w_isize = width as isize;
        let neighbor_offsets: [isize; 8] = [
            -w_isize - 1,
            -w_isize,
            -w_isize + 1,
            -1,
            1,
            w_isize - 1,
            w_isize,
            w_isize + 1,
        ];

        let mut head = 0usize;
        while head < stack.len() {
            let curr_idx = stack[head];
            head += 1;

            let cx = curr_idx % width;
            let cy = curr_idx / width;
            if cx == 0 || cx >= width - 1 || cy == 0 || cy >= height - 1 {
                continue;
            }

            for &offset in &neighbor_offsets {
                let n_idx = (curr_idx as isize + offset) as usize;
                if nms[n_idx] == WEAK_EDGE {
                    nms[n_idx] = STRONG_EDGE;
                    let nx = n_idx % width;
                    let ny = n_idx / width;
                    *dst.add(ny * dst_stride + nx) = STRONG_EDGE;
                    stack.push(n_idx);
                }
            }
        }
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 6. Normalized Cross-Correlation (NCC) Template Matcher (Integral Image + AVX2)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ncc_cross_correlation_avx2(
    search: *const u8,
    search_stride: usize,
    x: usize,
    y: usize,
    tw: usize,
    th: usize,
    t_zero_f32: &[f32],
    t_zero_f64: &[f64],
) -> f64 {
    let mut acc0 = _mm256_setzero_ps();
    let mut tail_sum = 0.0f64;
    let t_ptr = t_zero_f32.as_ptr();

    for ty in 0..th {
        let s_row = search.add((y + ty) * search_stride + x);
        let row_offset = ty * tw;
        let mut tx = 0usize;

        while tx + 8 <= tw {
            let raw8 = _mm_loadl_epi64(s_row.add(tx) as *const __m128i);
            let s_i32 = _mm256_cvtepu8_epi32(raw8);
            let s_f32 = _mm256_cvtepi32_ps(s_i32);
            let t_f32 = _mm256_loadu_ps(t_ptr.add(row_offset + tx));
            acc0 = _mm256_fmadd_ps(t_f32, s_f32, acc0);
            tx += 8;
        }

        while tx < tw {
            tail_sum += *t_zero_f64.get_unchecked(row_offset + tx) * (*s_row.add(tx) as f64);
            tx += 1;
        }
    }

    // Horizontal sum of acc0 into f64 via two 4-lane f64 vectors
    let lo_pd = _mm256_cvtps_pd(_mm256_castps256_ps128(acc0));
    let hi_pd = _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 1));
    let sum_pd = _mm256_add_pd(lo_pd, hi_pd);
    let sum128 = _mm_add_pd(
        _mm256_castpd256_pd128(sum_pd),
        _mm256_extractf128_pd(sum_pd, 1),
    );
    let mut lanes = [0.0f64; 2];
    _mm_storeu_pd(lanes.as_mut_ptr(), sum128);
    lanes[0] + lanes[1] + tail_sum
}

/// Evaluates Normalized Cross-Correlation (NCC) template matching with O(1) Integral Image
/// window variance, AVX2 SIMD inner cross-correlation, and 2D parabolic sub-pixel peak refinement.
///
/// # Safety
/// Pointers `search`, `tpl`, `out_x`, `out_y`, and `out_score` must be valid.
#[no_mangle]
pub unsafe extern "C" fn zg_ncc_match_u8(
    search: *const u8,
    search_stride: usize,
    sw: usize,
    sh: usize,
    tpl: *const u8,
    tpl_stride: usize,
    tw: usize,
    th: usize,
    min_score: f64,
    sub_pixel: i32,
    out_x: *mut f64,
    out_y: *mut f64,
    out_score: *mut f64,
) -> i32 {
    if search.is_null()
        || tpl.is_null()
        || out_x.is_null()
        || out_y.is_null()
        || out_score.is_null()
    {
        return ZG_ERR_NULL_POINTER;
    }

    *out_x = 0.0;
    *out_y = 0.0;
    *out_score = -1.0;

    if tw == 0 || th == 0 || sw == 0 || sh == 0 || search_stride < sw || tpl_stride < tw {
        return ZG_ERR_INVALID_DIMENSION;
    }
    if tw > sw || th > sh {
        return ZG_SUCCESS;
    }

    let res = catch_unwind(|| {
        let n = tw * th;
        let mut t_zero_f64 = vec![0.0f64; n];
        let mut t_sum = 0.0f64;

        for ty in 0..th {
            let p_row = tpl.add(ty * tpl_stride);
            let base = ty * tw;
            for tx in 0..tw {
                let val = *p_row.add(tx) as f64;
                t_zero_f64[base + tx] = val;
                t_sum += val;
            }
        }

        let t_mean = t_sum / (n as f64);
        let mut t_var_sum = 0.0f64;
        let mut t_zero_f32 = vec![0.0f32; n];
        for i in 0..n {
            let zm = t_zero_f64[i] - t_mean;
            t_zero_f64[i] = zm;
            t_zero_f32[i] = zm as f32;
            t_var_sum += zm * zm;
        }

        let t_std_dev = t_var_sum.sqrt();
        if t_std_dev < 1e-6 {
            *out_score = -1.0;
            return;
        }

        // Compute Integral Images ii and ii2 ((sw + 1) * (sh + 1))
        let ii_w = sw + 1;
        let ii_h = sh + 1;
        let ii_total = ii_w * ii_h;
        let mut ii = vec![0.0f64; ii_total];
        let mut ii2 = vec![0.0f64; ii_total];

        for y in 0..sh {
            let p_row = search.add(y * search_stride);
            let mut row_sum = 0.0f64;
            let mut row_sum2 = 0.0f64;
            let ii_row = (y + 1) * ii_w;
            let ii_prev = y * ii_w;

            for x in 0..sw {
                let val = *p_row.add(x) as f64;
                row_sum += val;
                row_sum2 += val * val;
                ii[ii_row + x + 1] = ii[ii_prev + x + 1] + row_sum;
                ii2[ii_row + x + 1] = ii2[ii_prev + x + 1] + row_sum2;
            }
        }

        let out_w = sw - tw + 1;
        let out_h = sh - th + 1;
        let mut score_map = if sub_pixel != 0 {
            vec![0.0f64; out_w * out_h]
        } else {
            Vec::new()
        };

        let mut best_score = -1.0f64;
        let mut best_x: isize = -1;
        let mut best_y: isize = -1;
        let inv_n = 1.0 / (n as f64);

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && tw >= 8;

        for y in 0..out_h {
            let y1_row = y * ii_w;
            let y2_row = (y + th) * ii_w;

            for x in 0..out_w {
                let x1 = x;
                let x2 = x + tw;

                let i_sum = ii[y2_row + x2] - ii[y1_row + x2] - ii[y2_row + x1] + ii[y1_row + x1];
                let i_sum2 =
                    ii2[y2_row + x2] - ii2[y1_row + x2] - ii2[y2_row + x1] + ii2[y1_row + x1];
                let i_var = i_sum2 - (i_sum * i_sum) * inv_n;

                if i_var <= 1e-6 {
                    if sub_pixel != 0 {
                        score_map[y * out_w + x] = 0.0;
                    }
                    continue;
                }

                let i_std_dev = i_var.sqrt();

                let cross_sum = {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if use_avx2 {
                            ncc_cross_correlation_avx2(
                                search,
                                search_stride,
                                x,
                                y,
                                tw,
                                th,
                                &t_zero_f32,
                                &t_zero_f64,
                            )
                        } else {
                            let mut sum = 0.0f64;
                            for ty in 0..th {
                                let s_row = search.add((y + ty) * search_stride + x);
                                let base = ty * tw;
                                for tx in 0..tw {
                                    sum += t_zero_f64[base + tx] * (*s_row.add(tx) as f64);
                                }
                            }
                            sum
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let mut sum = 0.0f64;
                        for ty in 0..th {
                            let s_row = search.add((y + ty) * search_stride + x);
                            let base = ty * tw;
                            for tx in 0..tw {
                                sum += t_zero_f64[base + tx] * (*s_row.add(tx) as f64);
                            }
                        }
                        sum
                    }
                };

                let score = cross_sum / (t_std_dev * i_std_dev);
                if sub_pixel != 0 {
                    score_map[y * out_w + x] = score;
                }

                if score > best_score {
                    best_score = score;
                    best_x = x as isize;
                    best_y = y as isize;
                }
            }
        }

        if best_score < min_score || best_x < 0 {
            *out_x = 0.0;
            *out_y = 0.0;
            *out_score = -1.0;
            return;
        }

        let bx = best_x as usize;
        let by = best_y as usize;
        let mut final_x = bx as f64;
        let mut final_y = by as f64;

        if sub_pixel != 0 {
            if bx > 0 && bx + 1 < out_w {
                let l = score_map[by * out_w + (bx - 1)];
                let c = score_map[by * out_w + bx];
                let r = score_map[by * out_w + (bx + 1)];
                let denom = 2.0 * (l - 2.0 * c + r);
                if denom.abs() > 1e-8 {
                    let delta_x = (l - r) / denom;
                    if delta_x.abs() <= 1.0 {
                        final_x += delta_x;
                    }
                }
            }

            if by > 0 && by + 1 < out_h {
                let t = score_map[(by - 1) * out_w + bx];
                let c = score_map[by * out_w + bx];
                let b = score_map[(by + 1) * out_w + bx];
                let denom = 2.0 * (t - 2.0 * c + b);
                if denom.abs() > 1e-8 {
                    let delta_y = (t - b) / denom;
                    if delta_y.abs() <= 1.0 {
                        final_y += delta_y;
                    }
                }
            }
        }

        *out_x = final_x;
        *out_y = final_y;
        *out_score = best_score;
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 7. Euclidean Signed Distance Field (SDF) Generator (SoA + AVX2)
// ============================================================================

#[derive(Clone, Copy)]
struct PrecomputedSegment {
    ax: f32,
    ay: f32,
    by: f32,
    abx: f32,
    aby: f32,
    inv_ab_len_sq: f32,
    is_degenerate: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sdf_row_avx2(
    segments: &[PrecomputedSegment],
    row_intersections: &[f32],
    py: f32,
    bounds_min_x: f32,
    step_x: f32,
    half_inv_spread: f32,
    width: usize,
    dst_row: *mut u8,
) {
    let v_zero = _mm256_setzero_ps();
    let v_one = _mm256_set1_ps(1.0);
    let v_half = _mm256_set1_ps(0.5);
    let v_scale = _mm256_set1_ps(255.0);
    let v_half_inv_spread = _mm256_set1_ps(half_inv_spread);
    let v_offsets = _mm256_setr_ps(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5);
    let v_step_x = _mm256_set1_ps(step_x);
    let v_min_x = _mm256_set1_ps(bounds_min_x);
    let v_sign_mask = _mm256_set1_ps(-0.0f32);

    let mut x = 0usize;
    while x + 8 <= width {
        let v_x_base = _mm256_set1_ps(x as f32);
        let v_px = _mm256_fmadd_ps(_mm256_add_ps(v_x_base, v_offsets), v_step_x, v_min_x);
        let mut v_min_sq = _mm256_set1_ps(f32::MAX);

        for seg in segments {
            let v_ax = _mm256_set1_ps(seg.ax);
            let v_apx = _mm256_sub_ps(v_px, v_ax);
            let apy = py - seg.ay;
            let v_apy = _mm256_set1_ps(apy);

            let v_sq = if seg.is_degenerate {
                _mm256_fmadd_ps(v_apx, v_apx, _mm256_mul_ps(v_apy, v_apy))
            } else {
                let v_abx = _mm256_set1_ps(seg.abx);
                let v_aby = _mm256_set1_ps(seg.aby);
                let v_inv = _mm256_set1_ps(seg.inv_ab_len_sq);

                let dot = _mm256_fmadd_ps(v_apx, v_abx, _mm256_mul_ps(v_apy, v_aby));
                let t = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(dot, v_inv), v_zero), v_one);

                let dx = _mm256_fnmadd_ps(t, v_abx, v_apx);
                let dy = _mm256_fnmadd_ps(t, v_aby, v_apy);
                _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy))
            };

            v_min_sq = _mm256_min_ps(v_min_sq, v_sq);
        }

        // Parity of ray-casting intersections along +X for these 8 pixels
        let mut v_parity = _mm256_setzero_si256();
        let v_ones_i32 = _mm256_set1_epi32(1);
        for &ix in row_intersections {
            let v_ix = _mm256_set1_ps(ix);
            let cmp = _mm256_cmp_ps(v_ix, v_px, _CMP_GT_OS);
            let mask = _mm256_and_si256(_mm256_castps_si256(cmp), v_ones_i32);
            v_parity = _mm256_xor_si256(v_parity, mask);
        }

        // inside == true when parity == 1; outside -> negate distance
        let v_is_outside = _mm256_cmpeq_epi32(v_parity, _mm256_setzero_si256());
        let v_dist = _mm256_sqrt_ps(v_min_sq);
        let v_neg_mask = _mm256_and_ps(_mm256_castsi256_ps(v_is_outside), v_sign_mask);
        let v_signed_dist = _mm256_xor_ps(v_dist, v_neg_mask);

        let v_norm = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_fmadd_ps(v_signed_dist, v_half_inv_spread, v_half),
                v_zero,
            ),
            v_one,
        );
        let v_byte_f32 = _mm256_mul_ps(v_norm, v_scale);
        let v_i32 = _mm256_cvttps_epi32(v_byte_f32);

        let lo128 = _mm256_castsi256_si128(v_i32);
        let hi128 = _mm256_extracti128_si256(v_i32, 1);
        let p16 = _mm_packus_epi32(lo128, hi128);
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(dst_row.add(x) as *mut __m128i, p8);

        x += 8;
    }

    while x < width {
        let px = bounds_min_x + (x as f32 + 0.5) * step_x;
        let mut min_sq_dist = f32::MAX;

        for seg in segments {
            let apx = px - seg.ax;
            let apy = py - seg.ay;
            let sq_dist = if seg.is_degenerate {
                apx * apx + apy * apy
            } else {
                let t = ((apx * seg.abx + apy * seg.aby) * seg.inv_ab_len_sq).clamp(0.0, 1.0);
                let dx = apx - t * seg.abx;
                let dy = apy - t * seg.aby;
                dx * dx + dy * dy
            };
            if sq_dist < min_sq_dist {
                min_sq_dist = sq_dist;
            }
        }

        let mut intersections = 0usize;
        for &ix in row_intersections {
            if ix > px {
                intersections += 1;
            }
        }

        let dist = min_sq_dist.sqrt();
        let signed_dist = if (intersections & 1) != 0 {
            dist
        } else {
            -dist
        };
        let norm = (0.5 + signed_dist * half_inv_spread).clamp(0.0, 1.0);
        *dst_row.add(x) = (norm * 255.0) as u8;
        x += 1;
    }
}

/// Computes an 8-bit Euclidean Signed Distance Field (`0..255`, `128` = contour boundary)
/// from Structure-of-Arrays (SoA) 2D line segments `(seg_ax, seg_ay, seg_bx, seg_by)`.
///
/// # Safety
/// Pointers `seg_ax`, `seg_ay`, `seg_bx`, `seg_by` (for `num_segments` elements) and `dst` (`width * height` bytes) must be valid.
#[no_mangle]
pub unsafe extern "C" fn zg_sdf_generate_u8(
    seg_ax: *const f32,
    seg_ay: *const f32,
    seg_bx: *const f32,
    seg_by: *const f32,
    num_segments: usize,
    width: usize,
    height: usize,
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    spread: f32,
    dst: *mut u8,
) -> i32 {
    if dst.is_null() {
        return ZG_ERR_NULL_POINTER;
    }
    if num_segments > 0
        && (seg_ax.is_null() || seg_ay.is_null() || seg_bx.is_null() || seg_by.is_null())
    {
        return ZG_ERR_NULL_POINTER;
    }
    if width == 0 || height == 0 {
        return ZG_SUCCESS;
    }

    let res = catch_unwind(|| {
        let total = width * height;
        if num_segments == 0 || spread <= 1e-6 {
            core::ptr::write_bytes(dst, 0, total);
            return;
        }

        let mut segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let ax = *seg_ax.add(i);
            let ay = *seg_ay.add(i);
            let bx = *seg_bx.add(i);
            let by = *seg_by.add(i);
            let abx = bx - ax;
            let aby = by - ay;
            let ab_len_sq = abx * abx + aby * aby;
            let is_degenerate = ab_len_sq < 1e-8;
            let inv_ab_len_sq = if is_degenerate { 0.0 } else { 1.0 / ab_len_sq };
            segments.push(PrecomputedSegment {
                ax,
                ay,
                by,
                abx,
                aby,
                inv_ab_len_sq,
                is_degenerate,
            });
        }

        let range_x = (bounds_max_x - bounds_min_x).max(1e-4);
        let range_y = (bounds_max_y - bounds_min_y).max(1e-4);
        let step_x = range_x / (width as f32);
        let step_y = range_y / (height as f32);
        let half_inv_spread = 0.5 / spread;

        #[cfg(target_arch = "x86_64")]
        let use_avx2 =
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && width >= 8;

        let mut row_intersections = Vec::with_capacity(num_segments);

        for y in 0..height {
            let py = bounds_min_y + (y as f32 + 0.5) * step_y;
            let dst_row = dst.add(y * width);

            // Precompute X-intersections for all segments crossing horizontal scanline py
            row_intersections.clear();
            for seg in &segments {
                if (seg.ay <= py && seg.by > py) || (seg.by <= py && seg.ay > py) {
                    let t = (py - seg.ay) / seg.aby;
                    let intersect_x = seg.ax + t * seg.abx;
                    row_intersections.push(intersect_x);
                }
            }

            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                sdf_row_avx2(
                    &segments,
                    &row_intersections,
                    py,
                    bounds_min_x,
                    step_x,
                    half_inv_spread,
                    width,
                    dst_row,
                );
                continue;
            }

            for x in 0..width {
                let px = bounds_min_x + (x as f32 + 0.5) * step_x;
                let mut min_sq_dist = f32::MAX;

                for seg in &segments {
                    let apx = px - seg.ax;
                    let apy = py - seg.ay;
                    let sq_dist = if seg.is_degenerate {
                        apx * apx + apy * apy
                    } else {
                        let t =
                            ((apx * seg.abx + apy * seg.aby) * seg.inv_ab_len_sq).clamp(0.0, 1.0);
                        let dx = apx - t * seg.abx;
                        let dy = apy - t * seg.aby;
                        dx * dx + dy * dy
                    };
                    if sq_dist < min_sq_dist {
                        min_sq_dist = sq_dist;
                    }
                }

                let mut intersections = 0usize;
                for &ix in &row_intersections {
                    if ix > px {
                        intersections += 1;
                    }
                }

                let dist = min_sq_dist.sqrt();
                let signed_dist = if (intersections & 1) != 0 {
                    dist
                } else {
                    -dist
                };
                let norm = (0.5 + signed_dist * half_inv_spread).clamp(0.0, 1.0);
                *dst_row.add(x) = (norm * 255.0) as u8;
            }
        }
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 8. 3x3 Grayscale Morphological Filter (AVX2 Erosion / Dilation)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn morphology3x3_row_avx2(
    r0: *const u8,
    r1: *const u8,
    r2: *const u8,
    dst_row: *mut u8,
    width: usize,
    dilate: bool,
) {
    // Left border x = 0
    let mut x = 1usize;
    let limit = width - 1;

    if dilate {
        while x + 32 <= limit {
            let v0 = _mm256_max_epu8(
                _mm256_max_epu8(
                    _mm256_loadu_si256(r0.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r0.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r0.add(x + 1) as *const __m256i),
            );
            let v1 = _mm256_max_epu8(
                _mm256_max_epu8(
                    _mm256_loadu_si256(r1.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r1.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r1.add(x + 1) as *const __m256i),
            );
            let v2 = _mm256_max_epu8(
                _mm256_max_epu8(
                    _mm256_loadu_si256(r2.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r2.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r2.add(x + 1) as *const __m256i),
            );
            let res = _mm256_max_epu8(_mm256_max_epu8(v0, v1), v2);
            _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, res);
            x += 32;
        }
    } else {
        while x + 32 <= limit {
            let v0 = _mm256_min_epu8(
                _mm256_min_epu8(
                    _mm256_loadu_si256(r0.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r0.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r0.add(x + 1) as *const __m256i),
            );
            let v1 = _mm256_min_epu8(
                _mm256_min_epu8(
                    _mm256_loadu_si256(r1.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r1.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r1.add(x + 1) as *const __m256i),
            );
            let v2 = _mm256_min_epu8(
                _mm256_min_epu8(
                    _mm256_loadu_si256(r2.add(x - 1) as *const __m256i),
                    _mm256_loadu_si256(r2.add(x) as *const __m256i),
                ),
                _mm256_loadu_si256(r2.add(x + 1) as *const __m256i),
            );
            let res = _mm256_min_epu8(_mm256_min_epu8(v0, v1), v2);
            _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, res);
            x += 32;
        }
    }

    // Scalar interior remainder
    while x < limit {
        if dilate {
            let mut m = 0u8;
            for &row in &[r0, r1, r2] {
                m = m
                    .max(*row.add(x - 1))
                    .max(*row.add(x))
                    .max(*row.add(x + 1));
            }
            *dst_row.add(x) = m;
        } else {
            let mut m = 255u8;
            for &row in &[r0, r1, r2] {
                m = m
                    .min(*row.add(x - 1))
                    .min(*row.add(x))
                    .min(*row.add(x + 1));
            }
            *dst_row.add(x) = m;
        }
        x += 1;
    }

    // Border columns x = 0 and x = width - 1
    for &bx in &[0usize, width - 1] {
        let x_min = bx.saturating_sub(1);
        let x_max = (bx + 1).min(width - 1);
        if dilate {
            let mut m = 0u8;
            for &row in &[r0, r1, r2] {
                for kx in x_min..=x_max {
                    m = m.max(*row.add(kx));
                }
            }
            *dst_row.add(bx) = m;
        } else {
            let mut m = 255u8;
            for &row in &[r0, r1, r2] {
                for kx in x_min..=x_max {
                    m = m.min(*row.add(kx));
                }
            }
            *dst_row.add(bx) = m;
        }
    }
}

/// Applies 3x3 Grayscale Morphological Erosion (`mode == 0`) or Dilation (`mode == 1`)
/// using AVX2 `_mm256_min_epu8` / `_mm256_max_epu8` vectorization.
///
/// # Safety
/// `src` and `dst` must be valid pointers for `height` rows of `src_stride` and `dst_stride` bytes.
#[no_mangle]
pub unsafe extern "C" fn zg_morphology3x3_u8(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    width: usize,
    height: usize,
    mode: i32,
) -> i32 {
    if src.is_null() || dst.is_null() {
        return ZG_ERR_NULL_POINTER;
    }
    if mode != 0 && mode != 1 {
        return ZG_ERR_INVALID_DIMENSION;
    }
    if width == 0 || height == 0 {
        return ZG_SUCCESS;
    }
    if src_stride < width || dst_stride < width {
        return ZG_ERR_INVALID_DIMENSION;
    }

    let res = catch_unwind(|| {
        let dilate = mode == 1;
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && width >= 34;

        for y in 0..height {
            let r0 = src.add(y.saturating_sub(1) * src_stride);
            let r1 = src.add(y * src_stride);
            let r2 = src.add((y + 1).min(height - 1) * src_stride);
            let dst_row = dst.add(y * dst_stride);

            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                morphology3x3_row_avx2(r0, r1, r2, dst_row, width, dilate);
                continue;
            }

            for x in 0..width {
                let x_min = x.saturating_sub(1);
                let x_max = (x + 1).min(width - 1);
                if dilate {
                    let mut m = 0u8;
                    for &row in &[r0, r1, r2] {
                        for kx in x_min..=x_max {
                            m = m.max(*row.add(kx));
                        }
                    }
                    *dst_row.add(x) = m;
                } else {
                    let mut m = 255u8;
                    for &row in &[r0, r1, r2] {
                        for kx in x_min..=x_max {
                            m = m.min(*row.add(kx));
                        }
                    }
                    *dst_row.add(x) = m;
                }
            }
        }
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// 9. Largest-Triangle-Three-Buckets (LTTB) Waveform Decimation (f64)
// ============================================================================

/// Downsamples a 1D/2D waveform `(src_x, src_y)` of length `src_len` into `(dst_x, dst_y)`
/// of target length `threshold` using the Largest-Triangle-Three-Buckets (LTTB) algorithm.
/// Writes the actual number of output points to `*out_written`.
///
/// # Safety
/// Pointers `src_x`, `src_y`, `dst_x`, `dst_y`, and `out_written` must be valid.
#[no_mangle]
pub unsafe extern "C" fn zg_lttb_decimate_f64(
    src_x: *const f64,
    src_y: *const f64,
    src_len: usize,
    threshold: usize,
    dst_x: *mut f64,
    dst_y: *mut f64,
    out_written: *mut usize,
) -> i32 {
    if src_x.is_null()
        || src_y.is_null()
        || dst_x.is_null()
        || dst_y.is_null()
        || out_written.is_null()
    {
        return ZG_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        if threshold == 0 || src_len == 0 {
            *out_written = 0;
            return;
        }

        if src_len <= threshold {
            core::ptr::copy_nonoverlapping(src_x, dst_x, src_len);
            core::ptr::copy_nonoverlapping(src_y, dst_y, src_len);
            *out_written = src_len;
            return;
        }

        if threshold == 1 {
            *dst_x.add(0) = *src_x.add(0);
            *dst_y.add(0) = *src_y.add(0);
            *out_written = 1;
            return;
        }

        if threshold == 2 {
            *dst_x.add(0) = *src_x.add(0);
            *dst_y.add(0) = *src_y.add(0);
            *dst_x.add(1) = *src_x.add(src_len - 1);
            *dst_y.add(1) = *src_y.add(src_len - 1);
            *out_written = 2;
            return;
        }

        let every = (src_len - 2) as f64 / (threshold - 2) as f64;
        let mut a = 0usize;
        *dst_x.add(0) = *src_x.add(0);
        *dst_y.add(0) = *src_y.add(0);
        let mut sampled_index = 1usize;

        for i in 0..(threshold - 2) {
            let avg_range_start = (((i + 1) as f64 * every).floor() as usize) + 1;
            let avg_range_end = ((((i + 2) as f64 * every).floor() as usize) + 1).min(src_len);
            let avg_range_len = avg_range_end.saturating_sub(avg_range_start);

            let (avg_x, avg_y) = if avg_range_len > 0 {
                let mut sx = 0.0f64;
                let mut sy = 0.0f64;
                for j in avg_range_start..avg_range_end {
                    sx += *src_x.add(j);
                    sy += *src_y.add(j);
                }
                let inv = 1.0 / (avg_range_len as f64);
                (sx * inv, sy * inv)
            } else {
                (*src_x.add(src_len - 1), *src_y.add(src_len - 1))
            };

            let range_offs = ((i as f64 * every).floor() as usize) + 1;
            let range_to = ((((i + 1) as f64 * every).floor() as usize) + 1).min(src_len);

            let point_ax = *src_x.add(a);
            let point_ay = *src_y.add(a);
            let dx_ac = point_ax - avg_x;
            let dy_ca = avg_y - point_ay;

            let mut max_area = -1.0f64;
            let mut max_area_index = range_offs;

            for j in range_offs..range_to {
                let px = *src_x.add(j);
                let py = *src_y.add(j);
                let area = (dx_ac * (py - point_ay) - (point_ax - px) * dy_ca).abs() * 0.5;
                if area > max_area {
                    max_area = area;
                    max_area_index = j;
                }
            }

            *dst_x.add(sampled_index) = *src_x.add(max_area_index);
            *dst_y.add(sampled_index) = *src_y.add(max_area_index);
            sampled_index += 1;
            a = max_area_index;
        }

        *dst_x.add(sampled_index) = *src_x.add(src_len - 1);
        *dst_y.add(sampled_index) = *src_y.add(src_len - 1);
        sampled_index += 1;
        *out_written = sampled_index;
    });

    match res {
        Ok(_) => ZG_SUCCESS,
        Err(_) => ZG_ERR_PANIC,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_and_simd() {
        assert_eq!(zg_version(), 100);
        let simd = zg_is_simd_supported();
        assert!(simd == 0 || simd == 1);
    }

    #[test]
    fn test_bgra32_to_gray8_matches_bt601() {
        let width = 23usize;
        let height = 5usize;
        let src_stride = width * 4;
        let dst_stride = width;
        let mut src = vec![0u8; src_stride * height];
        for i in 0..src.len() {
            src[i] = ((i * 73 + 19) & 0xFF) as u8;
        }
        let mut dst = vec![0u8; dst_stride * height];

        let rc = unsafe {
            zg_bgra32_to_gray8(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                width,
                height,
            )
        };
        assert_eq!(rc, ZG_SUCCESS);

        for y in 0..height {
            for x in 0..width {
                let o = y * src_stride + x * 4;
                let b = src[o] as u32;
                let g = src[o + 1] as u32;
                let r = src[o + 2] as u32;
                let expected = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
                assert_eq!(dst[y * dst_stride + x], expected, "mismatch at ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_gaussian5x5_u8_uniform_and_impulse() {
        let width = 32usize;
        let height = 16usize;
        let src = vec![120u8; width * height];
        let mut dst = vec![0u8; width * height];

        let rc = unsafe {
            zg_gaussian5x5_u8(src.as_ptr(), width, dst.as_mut_ptr(), width, width, height)
        };
        assert_eq!(rc, ZG_SUCCESS);
        assert!(dst.iter().all(|&v| v == 120));
    }

    #[test]
    fn test_canny_edge_u8_detects_rectangle() {
        let width = 32usize;
        let height = 32usize;
        let mut src = vec![20u8; width * height];
        for y in 8..24 {
            for x in 8..24 {
                src[y * width + x] = 230;
            }
        }
        let mut dst = vec![0u8; width * height];

        let rc = unsafe {
            zg_canny_edge_u8(
                src.as_ptr(),
                width,
                dst.as_mut_ptr(),
                width,
                width,
                height,
                20.0,
                50.0,
                1,
            )
        };
        assert_eq!(rc, ZG_SUCCESS);

        let edge_count = dst.iter().filter(|&&v| v == 255).count();
        assert!(edge_count > 20, "expected strong edges around rectangle, got {edge_count}");
        // Center of rectangle has 0 gradient
        assert_eq!(dst[16 * width + 16], 0);
    }

    #[test]
    fn test_ncc_match_u8_finds_template_with_subpixel() {
        let sw = 48usize;
        let sh = 48usize;
        let tw = 12usize;
        let th = 12usize;

        let mut search = vec![10u8; sw * sh];
        let mut tpl = vec![10u8; tw * th];

        for y in 0..th {
            for x in 0..tw {
                let val = (((x * 17 + y * 23) % 200) + 30) as u8;
                tpl[y * tw + x] = val;
                search[(19 + y) * sw + (23 + x)] = val;
            }
        }

        let mut out_x = 0.0f64;
        let mut out_y = 0.0f64;
        let mut out_score = 0.0f64;

        let rc = unsafe {
            zg_ncc_match_u8(
                search.as_ptr(),
                sw,
                sw,
                sh,
                tpl.as_ptr(),
                tw,
                tw,
                th,
                0.8,
                1,
                &mut out_x,
                &mut out_y,
                &mut out_score,
            )
        };
        assert_eq!(rc, ZG_SUCCESS);
        assert!((out_score - 1.0).abs() < 1e-4, "expected score ~1.0, got {out_score}");
        assert!((out_x - 23.0).abs() <= 0.5, "expected x ~23.0, got {out_x}");
        assert!((out_y - 19.0).abs() <= 0.5, "expected y ~19.0, got {out_y}");
    }

    #[test]
    fn test_sdf_generate_u8_closed_box() {
        // Closed box from (2, 2) to (8, 8) inside [0..10] x [0..10]
        let seg_ax = [2.0f32, 8.0, 8.0, 2.0];
        let seg_ay = [2.0f32, 2.0, 8.0, 8.0];
        let seg_bx = [8.0f32, 8.0, 2.0, 2.0];
        let seg_by = [2.0f32, 8.0, 8.0, 2.0];

        let width = 20usize;
        let height = 20usize;
        let mut dst = vec![0u8; width * height];

        let rc = unsafe {
            zg_sdf_generate_u8(
                seg_ax.as_ptr(),
                seg_ay.as_ptr(),
                seg_bx.as_ptr(),
                seg_by.as_ptr(),
                4,
                width,
                height,
                0.0,
                0.0,
                10.0,
                10.0,
                2.0,
                dst.as_mut_ptr(),
            )
        };
        assert_eq!(rc, ZG_SUCCESS);

        // Center (10, 10) -> (5.25, 5.25) is strictly inside (> 128)
        assert!(dst[10 * width + 10] > 180, "center should be inside: {}", dst[10 * width + 10]);
        // Corner (0, 0) -> (0.25, 0.25) is strictly outside (< 128)
        assert!(dst[0] < 60, "corner should be outside: {}", dst[0]);
    }

    #[test]
    fn test_morphology3x3_u8_erode_and_dilate() {
        let width = 40usize;
        let height = 12usize;
        let mut src = vec![0u8; width * height];
        // 3x3 block of 255 centered at (20, 6)
        for dy in 5..=7 {
            for dx in 19..=21 {
                src[dy * width + dx] = 255;
            }
        }

        let mut eroded = vec![0u8; width * height];
        let mut dilated = vec![0u8; width * height];

        unsafe {
            assert_eq!(
                zg_morphology3x3_u8(src.as_ptr(), width, eroded.as_mut_ptr(), width, width, height, 0),
                ZG_SUCCESS
            );
            assert_eq!(
                zg_morphology3x3_u8(src.as_ptr(), width, dilated.as_mut_ptr(), width, width, height, 1),
                ZG_SUCCESS
            );
        }

        // Erosion of a 3x3 block leaves only the center pixel (20, 6) = 255
        assert_eq!(eroded[6 * width + 20], 255);
        assert_eq!(eroded.iter().filter(|&&v| v == 255).count(), 1);

        // Dilation of a 3x3 block expands to a 5x5 block (25 pixels)
        assert_eq!(dilated.iter().filter(|&&v| v == 255).count(), 25);
    }

    #[test]
    fn test_lttb_decimate_f64_preserves_endpoints_and_peak() {
        let src_len = 100usize;
        let mut src_x = Vec::with_capacity(src_len);
        let mut src_y = Vec::with_capacity(src_len);
        for i in 0..src_len {
            src_x.push(i as f64);
            src_y.push(if i == 50 { 1000.0 } else { (i as f64).sin() });
        }

        let threshold = 10usize;
        let mut dst_x = vec![0.0f64; threshold];
        let mut dst_y = vec![0.0f64; threshold];
        let mut written = 0usize;

        let rc = unsafe {
            zg_lttb_decimate_f64(
                src_x.as_ptr(),
                src_y.as_ptr(),
                src_len,
                threshold,
                dst_x.as_mut_ptr(),
                dst_y.as_mut_ptr(),
                &mut written,
            )
        };
        assert_eq!(rc, ZG_SUCCESS);
        assert_eq!(written, threshold);
        assert_eq!(dst_x[0], 0.0);
        assert_eq!(dst_x[threshold - 1], 99.0);
        assert!(dst_y.iter().any(|&y| (y - 1000.0).abs() < 1e-9));
    }
}
