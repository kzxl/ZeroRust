//! Int8 quantization parameters, conversion functions, and activation kernels.

/// Quantization parameters defining scale $S$ and zero-point $Z$: $x \approx S \cdot (q - Z)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Floating-point scale factor.
    pub scale: f32,
    /// Integer zero-point offset.
    pub zero_point: i32,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
        }
    }
}

impl QuantParams {
    /// Creates a new symmetric quantization parameter set (zero_point = 0).
    pub const fn symmetric(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
        }
    }

    /// Creates an asymmetric quantization parameter set.
    pub const fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    /// Quantizes a 32-bit float value into a signed 8-bit integer:
    /// $q = \text{clamp}(\text{round}(x / S) + Z, -128, 127)$.
    pub fn quantize(&self, val: f32) -> i8 {
        let scaled = val / self.scale;
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5) as i32
        } else {
            (scaled - 0.5) as i32
        };
        let q = rounded + self.zero_point;
        if q > 127 {
            127
        } else if q < -128 {
            -128
        } else {
            q as i8
        }
    }

    /// Dequantizes a signed 8-bit integer back to a 32-bit float:
    /// $x = S \cdot (q - Z)$.
    pub fn dequantize(&self, q: i8) -> f32 {
        self.scale * (q as i32 - self.zero_point) as f32
    }
}

/// Requantizes a 32-bit accumulator into an 8-bit integer using fixed-point multiplier and right shift:
/// $q_{out} = \text{clamp}((acc \times multiplier) \gg shift, -128, 127)$.
pub fn requantize_i32(acc: i32, multiplier: i32, shift: u8) -> i8 {
    let scaled = (acc as i64 * multiplier as i64) >> shift;
    if scaled > 127 {
        127
    } else if scaled < -128 {
        -128
    } else {
        scaled as i8
    }
}

/// Clamped ReLU activation function for signed 8-bit integers:
/// $\text{ReLU}(x) = \max(0, x)$.
pub const fn relu_i8(x: i8) -> i8 {
    if x > 0 {
        x
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization_roundtrip() {
        let params = QuantParams::symmetric(0.1);
        let val = 5.4;
        let q = params.quantize(val);
        assert_eq!(q, 54);
        let recovered = params.dequantize(q);
        assert!((recovered - 5.4).abs() < 1e-4);
    }

    #[test]
    fn test_relu_i8() {
        assert_eq!(relu_i8(42), 42);
        assert_eq!(relu_i8(-10), 0);
        assert_eq!(relu_i8(0), 0);
    }
}
