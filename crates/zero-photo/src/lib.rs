#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use core::panic::PanicInfo;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub mod bw_filter;
pub mod exposure;
pub mod film_sim;
pub mod row_filter;
pub mod split_focus;

pub use bw_filter::{apply_bw_filter_luma, BwFilter};
pub use exposure::{zero_photo_calc_nd_exposure, NdFilter, RawHistogram, HIST_BINS};
pub use film_sim::{zero_film_sim_process_yuv422, FilmProfile, FilmStock};
pub use row_filter::{div_u32, rust_flt_get_crop_bounds, rust_flt_process_row};
pub use split_focus::{rust_split_get_circle_dx, rust_split_update_state, SplitFocusState, SPLIT_RETICLE_RADIUS};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_u32() {
        for n in [0u32, 1, 7, 255, 256, 65_535, u32::MAX] {
            for d in [1u32, 2, 3, 7, 255, 480, 65_535, u32::MAX] {
                assert_eq!(div_u32(n, d), n / d);
            }
        }
        assert_eq!(div_u32(100, 0), 0);
    }

    #[test]
    fn test_film_simulation_monochrome() {
        let stock = FilmStock::KodakTriX;
        let profile = stock.profile();
        let mut y0 = 100u8;
        let mut y1 = 150u8;
        let mut u = 200u8;
        let mut v = 50u8;

        stock.apply_yuv422_pair(&profile, &mut y0, &mut y1, &mut u, &mut v);
        assert_eq!(u, 128, "Tri-X chroma U must be 128 (neutral)");
        assert_eq!(v, 128, "Tri-X chroma V must be 128 (neutral)");
    }

    #[test]
    fn test_nd_calculator() {
        let filter = NdFilter::Nd64; // 6 stops
        assert_eq!(filter.calculate_shutter_ms(10), 640);
        assert_eq!(zero_photo_calc_nd_exposure(10, 6), 640);
    }

    #[test]
    fn test_crop_bounds() {
        let (mut ymin, mut ymax, mut xmin, mut xmax) = (0, 0, 0, 0);
        unsafe {
            rust_flt_get_crop_bounds(2, 720, 480, &mut ymin, &mut ymax, &mut xmin, &mut xmax);
        }
        assert_eq!((ymin, ymax, xmin, xmax), (83, 397, 0, 720));
    }

    #[test]
    fn test_bw_filter() {
        // Neutral gray stays neutral across all filters
        for f in [BwFilter::None, BwFilter::Yellow, BwFilter::Orange, BwFilter::Red, BwFilter::Green] {
            assert_eq!(apply_bw_filter_luma(128, 128, 128, f), 128);
        }

        // Blue sky (U > 128, low V) should be darkened by Red & Yellow filters
        let (y, u, v) = (100u8, 180u8, 110u8);
        let y_none = apply_bw_filter_luma(y, u, v, BwFilter::None);
        let y_yellow = apply_bw_filter_luma(y, u, v, BwFilter::Yellow);
        let y_red = apply_bw_filter_luma(y, u, v, BwFilter::Red);
        assert!(y_yellow <= y_none, "Yellow filter must darken blue sky");
        assert!(y_red < y_yellow, "Red filter must darken blue sky more aggressively than yellow");

        // Green foliage (low U, low V) should be brightened by Green filter
        let (yg, ug, vg) = (130u8, 100u8, 100u8);
        let yg_none = apply_bw_filter_luma(yg, ug, vg, BwFilter::None);
        let yg_green = apply_bw_filter_luma(yg, ug, vg, BwFilter::Green);
        assert!(yg_green >= yg_none, "Green filter must brighten green foliage");
    }

    #[test]
    fn test_split_focus() {
        let mut state = SplitFocusState::new();
        assert_eq!(state.shear_offset, 12);
        assert!(!state.is_locked);

        // Feed low gradient (blurry)
        for _ in 0..10 {
            state.update(500);
        }
        assert!(!state.is_locked);
        assert!(state.shear_offset > 0);

        // Feed sharp gradient (peak focus)
        for _ in 0..20 {
            state.update(12_000);
        }
        assert!(state.focus_score >= 88);
        assert!(state.is_locked);
        assert_eq!(state.shear_offset, 0, "Shear must snap to 0 on focus lock");

        // Test circle lookup table
        assert_eq!(rust_split_get_circle_dx(0), 56);
        assert_eq!(rust_split_get_circle_dx(56), 0);
        assert_eq!(rust_split_get_circle_dx(60), 0); // Out of bounds safe
    }
}
