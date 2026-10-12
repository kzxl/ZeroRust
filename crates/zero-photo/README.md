# zero-photo 📷

> Pure, deterministic computational photography, film simulations, tone curves, and real-time visual row filters for **ZeroRust**.

Part of the **ZeroUniverse** ecosystem.

## Features

- **Fujifilm-style Film Simulations**: Pure fixed-point integer color matrices and 1D LUT curves (Kodak Portra 400, Fuji Velvia 50, CineStill 800T, Kodak Tri-X 400, Classic Chrome, Kodachrome 64, Ilford HP5).
- **Embedded LiveView Row Filters**: Real-time cinema aspect-ratio cropping (XPan, CinemaScope, Univisium, 1:1, 4:3), Graduated ND, exposure compensation, and computational film recipe adjustments.
- **Zero-Allocation Division**: Software shift-subtract division (`div_u32`) for embedded ARM cores lacking hardware division.
- **Exposure Tools**: Optical Neutral Density exposure compensation (Bulb ND) and 14-bit RAW sensor histogram percentile analysis (ETTR).
- **`#![no_std]` Compatible**: Designed for bare-metal microcontrollers, camera DSPs, and high-performance desktop pipelines alike.

## Usage

```rust
use zero_photo::{FilmStock, NdFilter};

// Apply Film Stock to YUV422 pixel pair
let stock = FilmStock::FujiVelvia50;
let profile = stock.profile();
let (mut y0, mut y1, mut u, mut v) = (100, 150, 120, 140);
stock.apply_yuv422_pair(&profile, &mut y0, &mut y1, &mut u, &mut v);

// Calculate Bulb ND Exposure
let nd = NdFilter::Nd64; // 6 stops
let base_exposure_ms = 100;
let compensated_ms = nd.calculate_shutter_ms(base_exposure_ms); // 6400 ms
```

## License

Licensed under MIT. Authored by **Phong Vo & The ZeroUniverse Team**.
