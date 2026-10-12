//! Optical Contrast Filters for Black & White Monochrome Photography.
//!
//! Emulates physical colored glass filters placed in front of camera lenses:
//! - Yellow (#8): Darkens blue skies moderately, makes white clouds pop, natural tonal balance.
//! - Orange (#16): Deeper sky contrast than yellow, smooths skin tones for B&W portraits.
//! - Red (#25A): Dramatic black skies, extreme architectural and landscape contrast, haze cutting.
//! - Green (#11): Brightens foliage and vegetation, enhances tonal separation in nature.
//!
//! All calculations use fast integer fixed-point arithmetic (Q8) suitable for ARMv5TE embedded cores.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BwFilter {
    None = 0,
    Yellow = 1, // Wratten #8
    Orange = 2, // Wratten #16
    Red = 3,    // Wratten #25A
    Green = 4,  // Wratten #11
}

impl BwFilter {
    #[inline(always)]
    pub fn from_u32(val: u32) -> Self {
        match val {
            1 => BwFilter::Yellow,
            2 => BwFilter::Orange,
            3 => BwFilter::Red,
            4 => BwFilter::Green,
            _ => BwFilter::None,
        }
    }
}

#[inline(always)]
fn clamp_u8(val: i32) -> u8 {
    if val < 0 {
        0
    } else if val > 255 {
        255
    } else {
        val as u8
    }
}

/// Applies an optical contrast filter to a YUV422 pixel, returning the filtered luminance value.
///
/// Converts YUV to approximate RGB in Q8 fixed-point, computes the weighted optical transmission,
/// and returns the resulting luminance without dynamic memory allocation or division instructions.
#[inline(always)]
pub fn apply_bw_filter_luma(y: u8, u: u8, v: u8, filter: BwFilter) -> u8 {
    if filter == BwFilter::None {
        return y;
    }

    let u_diff = u as i32 - 128;
    let v_diff = v as i32 - 128;

    // Fixed-point YUV -> RGB conversion (Q8)
    let r = clamp_u8(y as i32 + ((359 * v_diff) >> 8)) as i32;
    let g = clamp_u8(y as i32 - ((88 * u_diff + 183 * v_diff) >> 8)) as i32;
    let b = clamp_u8(y as i32 + ((454 * u_diff) >> 8)) as i32;

    // Filter transmission weights (Sum = 256 for exact Q8 unity gain on neutral gray)
    let y_filt = match filter {
        BwFilter::None => y as i32,
        BwFilter::Yellow => (128 * r + 115 * g + 13 * b) >> 8,
        BwFilter::Orange => (174 * r + 72 * g + 10 * b) >> 8,
        BwFilter::Red =>    (230 * r + 21 * g + 5 * b) >> 8,
        BwFilter::Green =>  (38 * r + 192 * g + 26 * b) >> 8,
    };

    clamp_u8(y_filt)
}
