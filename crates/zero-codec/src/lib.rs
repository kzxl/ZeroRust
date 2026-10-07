//! # ZeroCodec
//!
//! High-speed tile-based screen compression and codec primitives for ZConn in ZeroRust:
//! - `ZeroTile` $32 \times 32$ pixel tile compression architecture
//! - High-throughput 64-bit SIMD XOR delta difference engine
//! - Zero-run byte-stream compression achieving up to 50:1 ratios on static/UI screens
//! - Planar YUV444p / YUV420p color space conversion
//! - Dynamic `BandwidthGovernor` and progressive `RollingIntraManager`
//! - MTU-safe datagram chunk packetizer

#![warn(missing_docs)]

pub mod color;
pub mod governor;
pub mod packet;
pub mod rle_lz4;
pub mod tile;
pub mod xor_simd;

pub use color::{bgra_to_yuv444p, yuv444p_to_bgra};
pub use governor::{BandwidthGovernor, RollingIntraManager};
pub use packet::{TileFragmentHeader, TilePacketizer, MAX_UDP_PAYLOAD_BYTES};
pub use rle_lz4::{compress_tile_rle, decompress_tile_rle};
pub use tile::{TileHeader, TileKind, ZeroTile, TILE_DIM, TILE_PIXELS, TILE_RAW_BYTES};
pub use xor_simd::{apply_tile_xor, compute_tile_xor, XorAnalysis};
