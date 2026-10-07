# zero-tinyai

Edge Neuromorphic & TinyAI engine for `ZeroRust` embedded systems and microcontrollers.

## Features
- **Ultra-Compact Footprint**: `< 16 KB` Flash, `< 4 KB` RAM, $< 500\,\mu\text{s}$ forward inference on ARM Cortex-M4.
- **Int8 Quantization**: Symmetric and asymmetric affine quantization with 32-bit accumulation and saturated requantization.
- **Quantized Layers**:
  - `QuantizedLinear`: Dense layer with fused bias and clamped ReLU activation.
  - `QuantizedConv1D`: 1D convolution over vibration waveforms and acoustic sensor data.
- **Predictive Maintenance AutoEncoder**: Compresses vibration FFT spectra into low-dimensional latent bottlenecks to detect mechanical wear and bearing failure weeks before catastrophic breakdown.
