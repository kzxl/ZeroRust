//! 1D Quantized Convolution kernel for temporal signal processing and spectral feature extraction.

use crate::quant::{relu_i8, requantize_i32};
use zero_core::error::{ZeroError, ZeroResult};

/// Quantized 1D Convolution kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizedConv1D<const KERNEL: usize> {
    /// 1D convolution filter kernel weights.
    pub weights: [i8; KERNEL],
    /// 32-bit bias.
    pub bias: i32,
    /// Fixed-point output multiplier.
    pub multiplier: i32,
    /// Right shift exponent.
    pub shift: u8,
}

impl<const KERNEL: usize> Default for QuantizedConv1D<KERNEL> {
    fn default() -> Self {
        Self {
            weights: [0i8; KERNEL],
            bias: 0,
            multiplier: 1 << 16,
            shift: 16,
        }
    }
}

impl<const KERNEL: usize> QuantizedConv1D<KERNEL> {
    /// Creates a new 1D conv kernel.
    pub const fn new(weights: [i8; KERNEL], bias: i32, multiplier: i32, shift: u8) -> Self {
        Self {
            weights,
            bias,
            multiplier,
            shift,
        }
    }

    /// Evaluates valid 1D convolution over `input` producing `output`.
    ///
    /// Requires `output.len() >= input.len() - KERNEL + 1`.
    pub fn convolve(&self, input: &[i8], output: &mut [i8], apply_relu: bool) -> ZeroResult<usize> {
        if input.len() < KERNEL {
            return Err(ZeroError::InvalidArgument);
        }
        let out_len = input.len() - KERNEL + 1;
        if output.len() < out_len {
            return Err(ZeroError::BufferOverflow);
        }

        for i in 0..out_len {
            let mut acc = self.bias;
            for k in 0..KERNEL {
                acc += (self.weights[k] as i32) * (input[i + k] as i32);
            }
            let mut val = requantize_i32(acc, self.multiplier, self.shift);
            if apply_relu {
                val = relu_i8(val);
            }
            output[i] = val;
        }

        Ok(out_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        // Moving average-like filter: [1, 2, 1]
        let conv = QuantizedConv1D::<3>::new([1, 2, 1], 0, 1 << 16, 16);
        let input = [10, 20, 30, 40];
        let mut out = [0i8; 2];

        // i=0: 1*10 + 2*20 + 1*30 = 80
        // i=1: 1*20 + 2*30 + 1*40 = 120
        let len = conv.convolve(&input, &mut out, false).unwrap();
        assert_eq!(len, 2);
        assert_eq!(out[0], 80);
        assert_eq!(out[1], 120);
    }
}
