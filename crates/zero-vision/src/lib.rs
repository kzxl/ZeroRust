//! # ZeroVision
//!
//! Lightweight embedded computer vision suite for ZeroRust:
//! - Const-generic zero-heap grayscale `ImageBuffer`
//! - Automatic Otsu and binary thresholding
//! - Sobel gradient edge detection and spatial Box blur
//! - Connected Component Labeling (CCL) and Blob Analysis
//! - Industrial fiducial crosshair and marker locator

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod blob;
pub mod filter;
pub mod image;
pub mod pattern;
pub mod threshold;

pub use blob::{Blob, BlobAnalyzer, BoundingBox};
pub use filter::{box_blur, sobel_filter};
pub use image::ImageBuffer;
pub use pattern::{CrosshairDetector, ImagePoint};
pub use threshold::{binarize, otsu_threshold};
