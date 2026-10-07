//! Fully connected quantized dense layer with 32-bit accumulation and saturated output.

use crate::quant::{relu_i8, requantize_i32};

/// Quantized Dense (Linear) Layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizedLinear<const IN_DIM: usize, const OUT_DIM: usize> {
    /// 8-bit signed weight matrix: OUT_DIM rows x IN_DIM columns.
    pub weights: [[i8; IN_DIM]; OUT_DIM],
    /// 32-bit bias vector.
    pub biases: [i32; OUT_DIM],
    /// Fixed-point output multiplier.
    pub multiplier: i32,
    /// Right shift exponent for requantization.
    pub shift: u8,
}

impl<const IN_DIM: usize, const OUT_DIM: usize> Default for QuantizedLinear<IN_DIM, OUT_DIM> {
    fn default() -> Self {
        Self {
            weights: [[0i8; IN_DIM]; OUT_DIM],
            biases: [0i32; OUT_DIM],
            multiplier: 1 << 16,
            shift: 16,
        }
    }
}

impl<const IN_DIM: usize, const OUT_DIM: usize> QuantizedLinear<IN_DIM, OUT_DIM> {
    /// Creates a new quantized linear layer.
    pub const fn new(
        weights: [[i8; IN_DIM]; OUT_DIM],
        biases: [i32; OUT_DIM],
        multiplier: i32,
        shift: u8,
    ) -> Self {
        Self {
            weights,
            biases,
            multiplier,
            shift,
        }
    }

    /// Evaluates the quantized forward pass with optional ReLU activation.
    pub fn forward(&self, input: &[i8; IN_DIM], output: &mut [i8; OUT_DIM], apply_relu: bool) {
        for r in 0..OUT_DIM {
            let mut acc: i32 = self.biases[r];
            for c in 0..IN_DIM {
                acc += (self.weights[r][c] as i32) * (input[c] as i32);
            }
            let mut out = requantize_i32(acc, self.multiplier, self.shift);
            if apply_relu {
                out = relu_i8(out);
            }
            output[r] = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward() {
        // 2 inputs, 1 output
        let mut layer = QuantizedLinear::<2, 1>::new([[2, 3]], [10], 1 << 16, 16);

        let input = [4, 5]; // 2*4 + 3*5 + 10 = 8 + 15 + 10 = 33
        let mut out = [0i8; 1];
        layer.forward(&input, &mut out, false);
        assert_eq!(out[0], 33);

        // With ReLU on negative result
        layer.biases = [-50]; // 23 - 50 = -27
        layer.forward(&input, &mut out, true);
        assert_eq!(out[0], 0);
    }
}
