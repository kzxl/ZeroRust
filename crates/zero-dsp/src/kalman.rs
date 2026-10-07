//! 1D Optimal Kalman Filter for real-time sensor fusion and state estimation.

/// 1D Linear Kalman Filter.
///
/// Models discrete process:
/// $x_k = x_{k-1} + w_k, \quad w_k \sim \mathcal{N}(0, Q)$
/// $z_k = x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanFilter1D {
    /// Estimated state $\hat{x}$.
    x: f64,
    /// Estimation error covariance $P$.
    p: f64,
    /// Process noise covariance $Q$.
    q: f64,
    /// Measurement noise covariance $R$.
    r: f64,
    /// Kalman gain $K$.
    k: f64,
}

impl KalmanFilter1D {
    /// Creates a new 1D Kalman Filter.
    ///
    /// * `initial_estimate`: Initial state value $\hat{x}_0$.
    /// * `initial_error`: Initial estimation variance $P_0$ (e.g. 1.0).
    /// * `process_noise`: Process variance $Q$ (e.g. 0.01).
    /// * `measurement_noise`: Sensor measurement noise variance $R$ (e.g. 0.1).
    pub const fn new(
        initial_estimate: f64,
        initial_error: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        Self {
            x: initial_estimate,
            p: initial_error,
            q: process_noise,
            r: measurement_noise,
            k: 0.0,
        }
    }

    /// Predicts state forward in time: $\hat{x}^- = \hat{x}$, $P^- = P + Q$.
    #[inline]
    pub fn predict(&mut self) {
        self.p += self.q;
    }

    /// Updates state with a new noisy measurement $z$:
    /// $K = \frac{P}{P + R}$
    /// $\hat{x} = \hat{x} + K (z - \hat{x})$
    /// $P = (1 - K) P$
    pub fn update(&mut self, measurement: f64) -> f64 {
        self.predict();
        self.k = self.p / (self.p + self.r);
        self.x += self.k * (measurement - self.x);
        self.p *= 1.0 - self.k;
        self.x
    }

    /// Returns the current optimal state estimate.
    #[inline]
    pub const fn estimate(&self) -> f64 {
        self.x
    }

    /// Returns the current estimation error covariance $P$.
    #[inline]
    pub const fn error_covariance(&self) -> f64 {
        self.p
    }

    /// Returns the last computed Kalman gain $K$.
    #[inline]
    pub const fn gain(&self) -> f64 {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_noise_reduction() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.001, 0.1);

        let true_val = 50.0;
        let mut estimated = 0.0;
        // Simulate constant signal with alternating noise (+1.0, -1.0)
        for i in 0..100 {
            let noise = if i % 2 == 0 { 1.0 } else { -1.0 };
            estimated = kf.update(true_val + noise);
        }

        // Kalman estimate should closely approximate true value (noise attenuated)
        assert!((estimated - true_val).abs() < 0.2);
    }
}
