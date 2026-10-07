use core::fmt;
use zero_core::ZeroError;

/// Error conditions encountered during virtual file system and security sandbox operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsError {
    /// Path attempts to traverse outside the jail root (e.g. `../../etc/passwd`).
    PathTraversalDenied,
    /// Path contains illegal null characters (`\0`).
    NullByteDetected,
    /// Path is empty or structurally malformed.
    InvalidPath,
    /// Path does not belong to the active jail sandbox.
    OutOfJailBounds,
    /// Underlying ZeroCore error.
    Core(ZeroError),
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathTraversalDenied => {
                write!(f, "Directory traversal denied: path escapes jail root")
            }
            Self::NullByteDetected => write!(f, "Illegal null byte detected in path"),
            Self::InvalidPath => write!(f, "Invalid or empty path"),
            Self::OutOfJailBounds => write!(f, "Path is outside the configured jail root"),
            Self::Core(err) => write!(f, "Underlying core error: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VfsError {}

impl From<ZeroError> for VfsError {
    fn from(err: ZeroError) -> Self {
        Self::Core(err)
    }
}
