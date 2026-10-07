//! Predictive maintenance Anomaly Detection AutoEncoder for vibration and acoustic emission monitoring.

use crate::dense::QuantizedLinear;
use crate::quant::QuantParams;

/// Quantized Anomaly Detection AutoEncoder operating on edge microcontrollers.
///
/// Learns the normal vibration envelope (FFT spectrum) of a spindle, gearbox, or motor bearing.
/// Reconstructs the spectrum and flags an anomaly when Mean Squared Error exceeds threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnomalyAutoEncoder<const INPUT_DIM: usize, const LATENT_DIM: usize> {
    /// Encoder layer compressing INPUT_DIM down to bottleneck LATENT_DIM.
    pub encoder: QuantizedLinear<INPUT_DIM, LATENT_DIM>,
    /// Decoder layer reconstructing LATENT_DIM back to INPUT_DIM.
    pub decoder: QuantizedLinear<LATENT_DIM, INPUT_DIM>,
    /// Quantization scaling parameters for input and output features.
    pub quant_params: QuantParams,
    /// Integer MSE anomaly threshold.
    pub threshold_mse: u32,
}

impl<const INPUT_DIM: usize, const LATENT_DIM: usize> AnomalyAutoEncoder<INPUT_DIM, LATENT_DIM> {
    /// Creates a new anomaly detection autoencoder.
    pub const fn new(
        encoder: QuantizedLinear<INPUT_DIM, LATENT_DIM>,
        decoder: QuantizedLinear<LATENT_DIM, INPUT_DIM>,
        quant_params: QuantParams,
        threshold_mse: u32,
    ) -> Self {
        Self {
            encoder,
            decoder,
            quant_params,
            threshold_mse,
        }
    }

    /// Evaluates forward reconstruction pass:
    /// $\text{latent} = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$
    /// $\hat{\mathbf{x}} = \mathbf{W}_d \text{latent} + \mathbf{b}_d$
    pub fn reconstruct(&self, input: &[i8; INPUT_DIM], reconstructed: &mut [i8; INPUT_DIM]) {
        let mut latent = [0i8; LATENT_DIM];
        self.encoder.forward(input, &mut latent, true);
        self.decoder.forward(&latent, reconstructed, false);
    }

    /// Computes discrete integer Mean Squared Error (MSE) loss between input and reconstructed signals:
    /// $\text{loss} = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - \hat{x}_i)^2$
    ///
    /// Returns `(loss_mse, is_anomaly)`.
    pub fn evaluate_anomaly(&self, input: &[i8; INPUT_DIM]) -> (u32, bool) {
        let mut reconstructed = [0i8; INPUT_DIM];
        self.reconstruct(input, &mut reconstructed);

        let mut sum_sq: u64 = 0;
        for i in 0..INPUT_DIM {
            let diff = (input[i] as i32) - (reconstructed[i] as i32);
            sum_sq += (diff * diff) as u64;
        }

        let mse = (sum_sq / (INPUT_DIM as u64)) as u32;
        let is_anomaly = mse > self.threshold_mse;
        (mse, is_anomaly)
    }

    /// Evaluates raw floating-point FFT magnitudes directly from `zero-dsp`.
    ///
    /// Quantizes inputs, evaluates integer autoencoder, and returns float MSE loss and anomaly flag.
    pub fn evaluate_fft_spectrum(&self, fft_magnitudes: &[f32; INPUT_DIM]) -> (f32, bool) {
        let mut q_input = [0i8; INPUT_DIM];
        for i in 0..INPUT_DIM {
            q_input[i] = self.quant_params.quantize(fft_magnitudes[i]);
        }

        let (int_mse, is_anomaly) = self.evaluate_anomaly(&q_input);
        let float_mse = (int_mse as f32) * self.quant_params.scale * self.quant_params.scale;

        (float_mse, is_anomaly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoencoder_anomaly_detection() {
        // Balanced identity autoencoder: 2 inputs -> 2 latent -> 2 outputs
        let enc_id = QuantizedLinear::<2, 2>::new([[1, 0], [0, 1]], [0, 0], 1 << 16, 16);
        let dec_id = QuantizedLinear::<2, 2>::new([[1, 0], [0, 1]], [0, 0], 1 << 16, 16);

        let ae_id = AnomalyAutoEncoder::new(enc_id, dec_id, QuantParams::symmetric(1.0), 5);

        // Normal pattern: [10, 20]
        let normal = [10, 20];
        let (loss_normal, is_anomaly_norm) = ae_id.evaluate_anomaly(&normal);
        assert_eq!(loss_normal, 0);
        assert!(!is_anomaly_norm);
    }
}
