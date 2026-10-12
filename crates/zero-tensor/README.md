# `zero-tensor` ⚡🦀

> **Sovereign, High-Performance Native Compute Substrate & Microkernel Acceleration for ZeroUniverse.**

Part of the [ZeroRust](https://github.com/kzxl/ZeroRust) framework, providing sub-millisecond C-ABI compute acceleration for [ZeroPlatform](https://github.com/kzxl/ZeroPlatform) (`ZeroTensor` and `ZeroCompute`).

---

## 🌟 Capabilities

- **Cache-Tiled FP32 GEMM**: 4x16 AVX2/FMA register-accumulated microkernels with strided layout support and multi-threaded row partitioning via `std::thread::scope`.
- **2D Block-Tiled FlashAttention-2**: CPU-optimized tiled online softmax ($B_c = 32$) with 4-way parallel dot products and GQA/MQA support.
- **Streaming On-The-Fly Quantized GEMV**: Direct memory-mapped Q4_0 and Q8_0 evaluation via AVX2 integer unpacking and FMA.
- **ABI-Stable C-FFI**: Standard C exports (`zero_tensor.dll` / `libzero_tensor.so`) for seamless, zero-copy C# P/Invoke integration.
