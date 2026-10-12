//! Real-time Film Simulation Engine for YUV422 LiveView and JPEG buffers.
//!
//! Provides integer-based Look-Up Tables (1D LUT) and Color Matrices:
//! - Luminance S-curves (highlight compression, shadow roll-off)
//! - Chroma matrix transformation (saturation scaling, hue rotation, split-toning)
//! - Zero memory allocations in frame rendering loops.

/// Film stock identifiers. Values 1..5 are shared with the C side (`struct film_recipe::base_stock`
/// in modules/silent/jpeg_enc.h, `live_flt.film_sim`, rust_flt) and must stay in sync with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FilmStock {
    Off = 0,
    KodakPortra400 = 1,
    FujiVelvia50 = 2,
    CineStill800T = 3,
    KodakTriX = 4,
    FujiClassicChrome = 5,
    KodakKodachrome64 = 6,
    IlfordHp5 = 7,
    SepiaVintage = 8,
}

#[derive(Clone, Copy)]
pub struct FilmProfile {
    pub sat_scale: i16,      // 256 = 1.0x saturation
    pub cross_talk: i16,     // Hue shift / color cross-mixing
    pub u_tint: i16,         // Blue/Yellow tint offset
    pub v_tint: i16,         // Red/Cyan tint offset
    pub contrast_slope: i16, // Midtone contrast curve multiplier (256 = 1.0)
    pub black_level: u8,     // Shadow lift or crush
}

impl FilmStock {
    pub fn profile(self) -> FilmProfile {
        match self {
            FilmStock::Off => FilmProfile {
                sat_scale: 256,
                cross_talk: 0,
                u_tint: 0,
                v_tint: 0,
                contrast_slope: 256,
                black_level: 0,
            },
            FilmStock::KodakPortra400 => FilmProfile {
                sat_scale: 230,      // Soft, natural pastel saturation
                cross_talk: -15,     // Soft skin tones
                u_tint: -8,          // Warm amber / golden bias
                v_tint: 12,          // Gentle rose/red warmth
                contrast_slope: 230, // Soft highlights roll-off
                black_level: 6,      // Slight shadow lift
            },
            FilmStock::FujiVelvia50 => FilmProfile {
                sat_scale: 340,      // High saturation
                cross_talk: 20,      // Vibrant greens and blues
                u_tint: 10,          // Vivid sky blue
                v_tint: 14,          // Deep warm reds
                contrast_slope: 300, // Punchy S-curve
                black_level: 0,      // Pure deep blacks
            },
            FilmStock::CineStill800T => FilmProfile {
                sat_scale: 256,
                cross_talk: 10,
                u_tint: 18,          // Cool tungsten shadows
                v_tint: 32,          // Distinct warm-red halation glow
                contrast_slope: 270,
                black_level: 4,
            },
            FilmStock::KodakTriX => FilmProfile {
                sat_scale: 0,        // Pure Monochrome
                cross_talk: 0,
                u_tint: 0,
                v_tint: 0,
                contrast_slope: 320, // Punchy street-photography S-curve
                black_level: 2,
            },
            FilmStock::FujiClassicChrome => FilmProfile {
                sat_scale: 195,      // Muted documentary color
                cross_talk: -10,
                u_tint: -4,
                v_tint: -6,
                contrast_slope: 290, // Hard shadows
                black_level: 0,
            },
            FilmStock::KodakKodachrome64 => FilmProfile {
                sat_scale: 280,
                cross_talk: -25,     // Iconic vintage Kodachrome palette
                u_tint: -12,         // Warm yellow cast
                v_tint: 24,          // Rich vermilion reds
                contrast_slope: 275,
                black_level: 3,
            },
            FilmStock::IlfordHp5 => FilmProfile {
                sat_scale: 0,        // Monochrome
                cross_talk: 0,
                u_tint: 0,
                v_tint: 0,
                contrast_slope: 240, // Smooth continuous tones, gentle contrast
                black_level: 8,      // Lifted matte blacks
            },
            FilmStock::SepiaVintage => FilmProfile {
                sat_scale: 0,
                cross_talk: 0,
                u_tint: -22,         // Golden Amber
                v_tint: 18,          // Warm Sepia Brown
                contrast_slope: 220,
                black_level: 12,
            },
        }
    }

    #[inline(always)]
    pub fn apply_yuv422_pair(self, profile: &FilmProfile, y0: &mut u8, y1: &mut u8, u: &mut u8, v: &mut u8) {
        if self == FilmStock::Off {
            return;
        }

        // 1. Process Luminance (Y0, Y1) with contrast curve
        let c_slope = profile.contrast_slope as i32;
        let b_level = profile.black_level as i32;

        let eval_luma = |y: u8| -> u8 {
            let dy = y as i32 - 128;
            let val = 128 + ((dy * c_slope) >> 8) + b_level;
            val.clamp(0, 255) as u8
        };

        *y0 = eval_luma(*y0);
        *y1 = eval_luma(*y1);

        // 2. Process Chrominance (U, V)
        if profile.sat_scale == 0 {
            // Monochrome or Tinted Monochrome
            *u = (128 + profile.u_tint).clamp(0, 255) as u8;
            *v = (128 + profile.v_tint).clamp(0, 255) as u8;
        } else {
            let du = *u as i32 - 128;
            let dv = *v as i32 - 128;

            let sat = profile.sat_scale as i32;
            let cross = profile.cross_talk as i32;

            // Saturation scaling with cross-talk color matrix
            let new_u = 128 + ((du * sat + dv * cross) >> 8) + profile.u_tint as i32;
            let new_v = 128 + ((dv * sat - du * cross) >> 8) + profile.v_tint as i32;

            *u = new_u.clamp(0, 255) as u8;
            *v = new_v.clamp(0, 255) as u8;
        }
    }
}

// ------------------------------------------------------------------------
// C ABI export for firmware modules and embedded loaders
// ------------------------------------------------------------------------
/// Applies a film stock in place to an interleaved UYVY buffer.
///
/// # Safety
/// `buf` must be null (no-op) or valid for reads and writes of `(num_pixels / 2) * 4` bytes, and no
/// other code (including DMA) may access that range for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn zero_film_sim_process_yuv422(
    stock_id: u32,
    buf: *mut u8,
    num_pixels: usize,
) {
    if buf.is_null() || stock_id == 0 || num_pixels < 2 {
        return;
    }

    let stock = match stock_id {
        1 => FilmStock::KodakPortra400,
        2 => FilmStock::FujiVelvia50,
        3 => FilmStock::CineStill800T,
        4 => FilmStock::KodakTriX,
        5 => FilmStock::FujiClassicChrome,
        6 => FilmStock::KodakKodachrome64,
        7 => FilmStock::IlfordHp5,
        8 => FilmStock::SepiaVintage,
        _ => return,
    };

    let profile = stock.profile();
    let num_pairs = num_pixels / 2;

    // SAFETY: the caller guarantees `buf` covers num_pairs * 4 bytes with exclusive access.
    let pixels = core::slice::from_raw_parts_mut(buf, num_pairs * 4);
    // YUV422 interleaved in Canon VRAM is: [U0, Y0, V0, Y1] (4 bytes per 2 pixels)
    for pair in pixels.chunks_exact_mut(4) {
        if let [u, y0, v, y1] = pair {
            stock.apply_yuv422_pair(&profile, y0, y1, u, v);
        }
    }
}
