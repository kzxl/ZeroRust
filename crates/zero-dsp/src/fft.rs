//! Power-of-two Radix-2 Cooley-Tukey Fast Fourier Transform (FFT) for vibration spectrum analysis.

use zero_core::error::{ZeroError, ZeroResult};

/// Complex number representation for FFT computations in no_std environments.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64 {
    /// Creates a new complex number.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Computes magnitude: $|z| = \sqrt{\text{re}^2 + \text{im}^2}$.
    pub fn magnitude(&self) -> f64 {
        let val = self.re * self.re + self.im * self.im;
        #[cfg(feature = "std")]
        {
            val.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            if val <= 0.0 {
                0.0
            } else {
                libm::sqrt(val)
            }
        }
    }
}

/// Radix-2 Fast Fourier Transform engine.
pub struct Radix2Fft;

impl Radix2Fft {
    /// Computes the in-place forward FFT of a complex slice.
    ///
    /// Slice length MUST be a power of two (e.g. 64, 128, 256, 512, 1024, 2048).
    pub fn transform(buffer: &mut [Complex64]) -> ZeroResult<()> {
        let n = buffer.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(ZeroError::InvalidArgument);
        }

        // In-place bit reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                buffer.swap(i, j);
            }
            let mut bit = n >> 1;
            while (j & bit) != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
        }

        // Cooley-Tukey decimation-in-time radix-2 butterflies
        let mut len = 2;
        while len <= n {
            #[cfg(feature = "std")]
            use std::f64::consts::PI;
            #[cfg(not(feature = "std"))]
            const PI: f64 = core::f64::consts::PI;

            let half = len / 2;
            let angle = -2.0 * PI / (len as f64);

            #[cfg(feature = "std")]
            let (w_re_step, w_im_step) = (angle.cos(), angle.sin());
            #[cfg(not(feature = "std"))]
            let (w_re_step, w_im_step) = (libm::cos(angle), libm::sin(angle));

            let mut i = 0;
            while i < n {
                let mut w_re = 1.0;
                let mut w_im = 0.0;

                for k in 0..half {
                    let u = buffer[i + k];
                    let v = buffer[i + k + half];

                    // v * w
                    let t_re = v.re * w_re - v.im * w_im;
                    let t_im = v.re * w_im + v.im * w_re;

                    buffer[i + k] = Complex64::new(u.re + t_re, u.im + t_im);
                    buffer[i + k + half] = Complex64::new(u.re - t_re, u.im - t_im);

                    // Update twiddle factor
                    let next_w_re = w_re * w_re_step - w_im * w_im_step;
                    let next_w_im = w_re * w_im_step + w_im * w_re_step;
                    w_re = next_w_re;
                    w_im = next_w_im;
                }
                i += len;
            }
            len <<= 1;
        }

        Ok(())
    }

    /// Computes the magnitude spectrum of the complex FFT output into a preallocated slice.
    pub fn compute_magnitudes(fft_output: &[Complex64], magnitudes: &mut [f64]) -> ZeroResult<()> {
        let half = fft_output.len() / 2;
        if magnitudes.len() < half {
            return Err(ZeroError::BufferOverflow);
        }

        for (i, item) in fft_output.iter().take(half).enumerate() {
            magnitudes[i] = item.magnitude();
        }

        Ok(())
    }

    /// Finds the dominant peak frequency index in the magnitude spectrum (ignoring DC at index 0).
    pub fn find_peak_bin(magnitudes: &[f64]) -> usize {
        let mut max_val = 0.0;
        let mut peak_idx = 1;

        for (i, &mag) in magnitudes.iter().enumerate().skip(1) {
            if mag > max_val {
                max_val = mag;
                peak_idx = i;
            }
        }

        peak_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_peak_detection() {
        const N: usize = 64;
        let mut buf = [Complex64::default(); N];

        // Generate a 4-cycle sine wave across 64 samples -> peak should be at bin 4
        #[cfg(feature = "std")]
        use std::f64::consts::PI;
        #[cfg(not(feature = "std"))]
        const PI: f64 = core::f64::consts::PI;

        for (i, item) in buf.iter_mut().enumerate() {
            let t = i as f64 / N as f64;
            item.re = (2.0 * PI * 4.0 * t).sin();
        }

        Radix2Fft::transform(&mut buf).unwrap();

        let mut mags = [0.0; N / 2];
        Radix2Fft::compute_magnitudes(&buf, &mut mags).unwrap();

        let peak_bin = Radix2Fft::find_peak_bin(&mags);
        assert_eq!(peak_bin, 4);
    }
}
