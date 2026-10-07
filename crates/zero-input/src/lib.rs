//! # ZeroInput
//!
//! Synthetic input injection and cross-platform peripheral emulation for ZConn in ZeroRust:
//! - Standardized `MouseEvent` (absolute $[0.0, 1.0]$ normalization & relative delta modes)
//! - USB HID Usage Codes (Page 0x07) scancode mapping for layout independence
//! - `SafetyGuard` anti-stuck modifier state machine with emergency key release
//! - `ClipboardSyncManager` with FNV-1a hash-based echo suppression
//! - `InputInjector` trait and `MockInjector` for headless testing

#![warn(missing_docs)]

pub mod clipboard;
pub mod events;
pub mod guard;
pub mod injector;
pub mod scancodes;

pub use clipboard::{compute_fnv1a_hash, ClipboardEvent, ClipboardFormat, ClipboardSyncManager};
pub use events::{ElementState, KeyboardEvent, MouseButton, MouseEvent};
pub use guard::SafetyGuard;
pub use injector::{InputError, InputInjector, MockInjector};
pub use scancodes::{hid, hid_to_windows_vk};
