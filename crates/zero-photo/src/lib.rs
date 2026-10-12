#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use core::panic::PanicInfo;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub mod exposure;
pub mod film_sim;
pub mod row_filter;

pub use exposure::{zero_photo_calc_nd_exposure, NdFilter, RawHistogram, HIST_BINS};
pub use film_sim::{zero_film_sim_process_yuv422, FilmProfile, FilmStock};
pub use row_filter::{div_u32, rust_flt_get_crop_bounds, rust_flt_process_row};

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
}
