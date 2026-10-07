//! # Zero-VFS (`zero-vfs`)
//!
//! Sovereign Virtual File System, Security Jail, and Path Sanitization
//! Engine for Web Panels (ZPanl) and Embedded Edge Controllers.
//!
//! ## Modules
//! - [`error`]: Virtual file system errors and path traversal denial taxonomy.
//! - [`path`]: Path normalization, null-byte rejection, and traversal jail protection.
//! - [`entry`]: File entry descriptor, metadata, and size formatting.
//! - [`mime`]: Zero-allocation MIME type resolution and text/binary detection.
//! - [`atomic`]: Atomic file update naming and staging lifecycle.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod atomic;
pub mod entry;
pub mod error;
pub mod mime;
pub mod path;

pub use atomic::AtomicPathBuilder;
pub use entry::{VfsEntry, VfsFileType};
pub use error::VfsError;
pub use mime::{detect_mime, is_archive, is_text_editable};
pub use path::JailSandbox;
