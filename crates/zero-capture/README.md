# `zero-capture` ⚡🖥️

> Cross-platform ultra-low-latency screen acquisition, cacheline-aligned frame buffer pools, and SIMD damage detection for ZConn in ZeroRust.

## Features
- **Zero-Allocation Video Loop**: Pre-allocated `FrameBufferPool` eliminates all runtime heap allocations.
- **Cacheline Aligned**: 64-byte alignment allows direct AVX2 / NEON vector loads without boundary faults.
- **Damage & Dirty-Rect Tracking**: Divides display into 32x32 tiles, isolating modified regions.
- **Deterministic Headless Testing**: Built-in `MockCapturer` enables CI test suites without display hardware.
