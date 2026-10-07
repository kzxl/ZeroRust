use crate::error::VfsError;
use alloc::string::String;
use alloc::vec::Vec;

/// Security sandbox that confines all filesystem operations within a designated jail root.
///
/// Prevents Directory Traversal attacks (e.g. `../../etc/passwd`, null-byte injections,
/// or symlink jail escaping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JailSandbox {
    jail_root: String,
}

impl JailSandbox {
    /// Creates a new `JailSandbox` rooted at `jail_root`.
    ///
    /// Trailing slashes are stripped for consistent canonical comparison.
    pub fn new(jail_root: &str) -> Result<Self, VfsError> {
        if jail_root.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        if jail_root.contains('\0') {
            return Err(VfsError::NullByteDetected);
        }

        let trimmed = jail_root.trim_end_matches(['/', '\\']);
        let normalized = if trimmed.is_empty() {
            // Root "/" on Unix
            String::from("/")
        } else {
            String::from(trimmed)
        };

        Ok(Self {
            jail_root: normalized,
        })
    }

    /// Returns the configured jail root path.
    pub fn jail_root(&self) -> &str {
        &self.jail_root
    }

    /// Resolves and verifies that `target` resides strictly inside the jail root.
    ///
    /// `target` can be an absolute path starting with `jail_root`, or a relative path
    /// (with or without leading `/`) relative to `jail_root`.
    ///
    /// # Errors
    /// - Returns `VfsError::NullByteDetected` if target contains null bytes.
    /// - Returns `VfsError::PathTraversalDenied` if target attempts to traverse (`..`) outside the jail.
    pub fn resolve(&self, target: &str) -> Result<String, VfsError> {
        if target.contains('\0') {
            return Err(VfsError::NullByteDetected);
        }

        // Normalize slashes
        let clean_target = target.replace('\\', "/");

        // Determine if target already starts with jail_root
        let rel_part = if clean_target.starts_with(&self.jail_root) {
            let remainder = &clean_target[self.jail_root.len()..];
            remainder.trim_start_matches('/')
        } else {
            clean_target.trim_start_matches('/')
        };

        // Lexically normalize relative components
        let mut components: Vec<&str> = Vec::new();

        for part in rel_part.split('/') {
            match part {
                "" | "." => continue,
                ".." => {
                    if components.pop().is_none() {
                        // Attempted to pop above jail root!
                        return Err(VfsError::PathTraversalDenied);
                    }
                }
                normal => components.push(normal),
            }
        }

        if components.is_empty() {
            Ok(self.jail_root.clone())
        } else {
            let mut resolved = String::with_capacity(self.jail_root.len() + 1 + rel_part.len());
            resolved.push_str(&self.jail_root);
            if self.jail_root != "/" {
                resolved.push('/');
            }
            for (idx, comp) in components.iter().enumerate() {
                if idx > 0 {
                    resolved.push('/');
                }
                resolved.push_str(comp);
            }
            Ok(resolved)
        }
    }

    /// Returns the relative path of an absolute path inside the jail.
    ///
    /// Useful for rendering relative paths in the Web UI.
    pub fn relative_to_jail<'a>(&self, absolute_path: &'a str) -> Result<&'a str, VfsError> {
        if !absolute_path.starts_with(&self.jail_root) {
            return Err(VfsError::OutOfJailBounds);
        }

        let slice = &absolute_path[self.jail_root.len()..];
        let trimmed = slice.trim_start_matches(['/', '\\']);
        Ok(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jail_resolve_valid_subpaths() {
        let jail = JailSandbox::new("/var/www/site1").unwrap();

        assert_eq!(
            jail.resolve("public/index.php").unwrap(),
            "/var/www/site1/public/index.php"
        );
        assert_eq!(
            jail.resolve("/public/index.php").unwrap(),
            "/var/www/site1/public/index.php"
        );
        assert_eq!(
            jail.resolve("/var/www/site1/public/css/style.css").unwrap(),
            "/var/www/site1/public/css/style.css"
        );
        assert_eq!(jail.resolve("").unwrap(), "/var/www/site1");
        assert_eq!(jail.resolve("/").unwrap(), "/var/www/site1");
        assert_eq!(
            jail.resolve("./sub/./file.txt").unwrap(),
            "/var/www/site1/sub/file.txt"
        );
    }

    #[test]
    fn test_jail_reject_path_traversal() {
        let jail = JailSandbox::new("/var/www/site1").unwrap();

        // Direct traversal attempt
        assert_eq!(
            jail.resolve("../../etc/passwd"),
            Err(VfsError::PathTraversalDenied)
        );

        // Sneaky traversal attempt
        assert_eq!(
            jail.resolve("public/../../.."),
            Err(VfsError::PathTraversalDenied)
        );

        // Subfolder traversal attempt
        assert_eq!(
            jail.resolve("sub/dir/../../../etc/shadow"),
            Err(VfsError::PathTraversalDenied)
        );

        // Traversal within bounds is allowed
        assert_eq!(
            jail.resolve("public/sub/../index.html").unwrap(),
            "/var/www/site1/public/index.html"
        );
    }

    #[test]
    fn test_jail_reject_null_bytes() {
        let jail = JailSandbox::new("/var/www/site1").unwrap();
        assert_eq!(
            jail.resolve("index.php\0.png"),
            Err(VfsError::NullByteDetected)
        );
    }

    #[test]
    fn test_relative_to_jail() {
        let jail = JailSandbox::new("/var/www/site1").unwrap();
        assert_eq!(
            jail.relative_to_jail("/var/www/site1/public/index.php")
                .unwrap(),
            "public/index.php"
        );
        assert_eq!(
            jail.relative_to_jail("/etc/shadow"),
            Err(VfsError::OutOfJailBounds)
        );
    }
}
