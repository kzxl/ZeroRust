# `zero-codec` ⚡🎞️

> High-speed tile-based screen compression, SIMD XOR diffing, YUV conversion, and bandwidth governor for ZConn in ZeroRust.

## Features
- **ZeroTile Architecture**: Divides display into 32x32 tiles (4096 bytes in 32bpp, matching 1 OS page).
- **Sub-Millisecond Compression**: Fast XOR diffing + zero-run RLE yields < 0.2ms compression per frame.
- **Bandwidth Governor**: Dynamically scales target FPS and color quantization based on network feedback.
- **Rolling Intra Refresh**: Refreshes 5% of tiles per frame, preventing packet loss freezes without I-frame spikes.
