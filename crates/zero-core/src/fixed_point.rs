//! Fixed-point mathematics primitives for deterministic real-time motion and embedded compute.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// Fixed-point signed 32-bit number with 16 fractional bits (Q16.16).
///
/// Range: [-32768.0, 32767.99998] with resolution of 1/65536 (~0.000015).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q16_16(pub i32);

impl Q16_16 {
    const FRACTION_BITS: u32 = 16;
    const SCALE: f32 = 65536.0;

    /// Zero value.
    pub const ZERO: Self = Self(0);
    /// One value.
    pub const ONE: Self = Self(1 << Self::FRACTION_BITS);

    /// Constructs from raw internal integer representation.
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    /// Returns the raw internal integer.
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Constructs a Q16.16 from an integer.
    #[inline]
    pub const fn from_int(val: i16) -> Self {
        Self((val as i32) << Self::FRACTION_BITS)
    }

    /// Constructs a Q16.16 from an f32.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        Self((val * Self::SCALE) as i32)
    }

    /// Converts Q16.16 to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE
    }

    /// Computes absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Add for Q16_16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for Q16_16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for Q16_16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let product = (self.0 as i64) * (rhs.0 as i64);
        Self((product >> Self::FRACTION_BITS) as i32)
    }
}

impl Div for Q16_16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            if self.0 >= 0 {
                return Self(i32::MAX);
            } else {
                return Self(i32::MIN);
            }
        }
        let dividend = (self.0 as i64) << Self::FRACTION_BITS;
        Self((dividend / (rhs.0 as i64)) as i32)
    }
}

impl Neg for Q16_16 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_16_arithmetic() {
        let a = Q16_16::from_f32(1.5);
        let b = Q16_16::from_f32(2.25);

        let sum = a + b;
        assert!((sum.to_f32() - 3.75).abs() < 0.001);

        let diff = b - a;
        assert!((diff.to_f32() - 0.75).abs() < 0.001);

        let prod = a * b;
        assert!((prod.to_f32() - 3.375).abs() < 0.001);

        let div = b / a;
        assert!((div.to_f32() - 1.5).abs() < 0.001);
    }
}
