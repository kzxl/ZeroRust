use crate::error::VfsError;
use alloc::format;
use alloc::string::String;

/// Safe atomic file replacement path generator.
///
/// Ensures temporary and backup files reside in the same parent directory
/// as the target to guarantee atomic POSIX `rename(2)` operations without
/// cross-device link failures (EXDEV).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicPathBuilder {
    target_path: String,
}

impl AtomicPathBuilder {
    /// Creates an `AtomicPathBuilder` for the specified target path.
    pub fn new(target_path: impl Into<String>) -> Result<Self, VfsError> {
        let path = target_path.into();
        if path.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        if path.contains('\0') {
            return Err(VfsError::NullByteDetected);
        }
        Ok(Self { target_path: path })
    }

    /// Returns the target destination path.
    pub fn target_path(&self) -> &str {
        &self.target_path
    }

    /// Generates the sibling temporary staging path (e.g. `<target>.zpanl.tmp`).
    pub fn temp_path(&self) -> String {
        format!("{}.zpanl.tmp", self.target_path)
    }

    /// Generates the sibling backup path (e.g. `<target>.zpanl.bak`).
    pub fn backup_path(&self) -> String {
        format!("{}.zpanl.bak", self.target_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_path_generation() {
        let builder = AtomicPathBuilder::new("/var/www/site1/public/wp-config.php").unwrap();

        assert_eq!(builder.target_path(), "/var/www/site1/public/wp-config.php");
        assert_eq!(
            builder.temp_path(),
            "/var/www/site1/public/wp-config.php.zpanl.tmp"
        );
        assert_eq!(
            builder.backup_path(),
            "/var/www/site1/public/wp-config.php.zpanl.bak"
        );
    }

    #[test]
    fn test_invalid_target_path() {
        assert_eq!(AtomicPathBuilder::new(""), Err(VfsError::InvalidPath));
        assert_eq!(
            AtomicPathBuilder::new("file\0name.php"),
            Err(VfsError::NullByteDetected)
        );
    }
}
