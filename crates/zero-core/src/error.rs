//! Common error definitions for ZeroRust core primitives.

use core::fmt;

/// Standard error taxonomy for ZeroRust systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroError {
    /// Bounded buffer is full and cannot accept additional elements.
    BufferFull,
    /// Bounded buffer is empty.
    BufferEmpty,
    /// Attempted to write past the end of a slice.
    BufferOverflow,
    /// Slice or span ended unexpectedly before reading requested data.
    UnexpectedEndOfBuffer,
    /// Provided argument is invalid or out of range.
    InvalidArgument,
    /// Mathematical domain error or division by zero.
    MathError,
    /// Hardware or fieldbus communication failure.
    HardwareFault,
    /// Operation timed out.
    Timeout,
    /// Unsupported protocol feature or drive mode.
    Unsupported,
}

impl fmt::Display for ZeroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull => write!(f, "Buffer is full"),
            Self::BufferEmpty => write!(f, "Buffer is empty"),
            Self::BufferOverflow => write!(f, "Buffer overflow"),
            Self::UnexpectedEndOfBuffer => write!(f, "Unexpected end of buffer"),
            Self::InvalidArgument => write!(f, "Invalid argument"),
            Self::MathError => write!(f, "Mathematical or numerical calculation error"),
            Self::HardwareFault => write!(f, "Hardware or communication fault"),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::Unsupported => write!(f, "Unsupported operation or mode"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ZeroError {}

/// Unified result type for zero-core.
pub type ZeroResult<T> = Result<T, ZeroError>;
