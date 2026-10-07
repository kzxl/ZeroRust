//! Real-time digital filtering primitives: Moving Average and 2nd-Order Butterworth IIR.

/// $O(1)$ Moving Average filter with compile-time bounded window size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingAverage<const WINDOW_SIZE: usize> {
    buffer: [f64; WINDOW_SIZE],
    index: usize,
    count: usize,
    sum: f64,
}

impl<const WINDOW_SIZE: usize> Default for MovingAverage<WINDOW_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WINDOW_SIZE: usize> MovingAverage<WINDOW_SIZE> {
    /// Creates a new moving average filter.
    pub const fn new() -> Self {
        assert!(WINDOW_SIZE > 0, "Window size must be greater than zero");
        Self {
            buffer: [0.0; WINDOW_SIZE],
            index: 0,
            count: 0,
            sum: 0.0,
        }
    }

    /// Feeds a new input sample and returns the current running average.
    pub fn update(&mut self, sample: f64) -> f64 {
        if self.count < WINDOW_SIZE {
            self.sum += sample;
            self.buffer[self.index] = sample;
            self.count += 1;
            self.index = (self.index + 1) % WINDOW_SIZE;
            self.sum / self.count as f64
        } else {
            self.sum -= self.buffer[self.index];
            self.sum += sample;
            self.buffer[self.index] = sample;
            self.index = (self.index + 1) % WINDOW_SIZE;
            self.sum / WINDOW_SIZE as f64
        }
    }

    /// Resets the filter to initial state.
    pub fn reset(&mut self) {
        self.buffer = [0.0; WINDOW_SIZE];
        self.index = 0;
        self.count = 0;
        self.sum = 0.0;
    }
}

/// 2nd-Order Butterworth Low-Pass IIR Filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButterworthLowPass {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl ButterworthLowPass {
    /// Configures a 2nd-order low-pass Butterworth filter.
    ///
    /// * `cutoff_hz`: Cutoff frequency (-3dB point) in Hertz.
    /// * `sample_rate_hz`: Sampling frequency in Hertz.
    pub fn new(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        #[cfg(feature = "std")]
        use std::f64::consts::PI;
        #[cfg(not(feature = "std"))]
        const PI: f64 = core::f64::consts::PI;

        let fc = cutoff_hz.min(sample_rate_hz * 0.499);

        #[cfg(feature = "std")]
        let (ita, q) = {
            let ita = 1.0 / (PI * fc / sample_rate_hz).tan();
            let q = 2.0_f64.sqrt();
            (ita, q)
        };
        #[cfg(not(feature = "std"))]
        let (ita, q) = {
            let ita = 1.0 / libm::tan(PI * fc / sample_rate_hz);
            let q = libm::sqrt(2.0);
            (ita, q)
        };

        let b0 = 1.0 / (1.0 + q * ita + ita * ita);
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (ita * ita - 1.0) * b0;
        let a2 = -(1.0 - q * ita + ita * ita) * b0;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Filters a new input sample using Direct Form I difference equations.
    pub fn update(&mut self, sample: f64) -> f64 {
        let y = self.b0 * sample
            + self.b1 * self.x1
            + self.b2 * self.x2
            + self.a1 * self.y1
            + self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = sample;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Resets historical filter states.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let mut filter: MovingAverage<4> = MovingAverage::new();
        assert_eq!(filter.update(10.0), 10.0);
        assert_eq!(filter.update(20.0), 15.0);
        assert_eq!(filter.update(30.0), 20.0);
        assert_eq!(filter.update(40.0), 25.0);
        assert_eq!(filter.update(50.0), 35.0); // (20 + 30 + 40 + 50) / 4 = 35
    }

    #[test]
    fn test_butterworth_dc_gain() {
        let mut filter = ButterworthLowPass::new(10.0, 1000.0);
        let mut y = 0.0;
        for _ in 0..500 {
            y = filter.update(100.0); // Constant DC step
        }
        // DC gain of Butterworth lowpass should converge to unity (100.0)
        assert!((y - 100.0).abs() < 0.1);
    }
}
