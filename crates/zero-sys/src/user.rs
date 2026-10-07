//! POSIX user isolation, permissions, and security validation for web roots.

use zero_core::error::{ZeroError, ZeroResult};

/// POSIX file permission mode bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PosixPermissions {
    /// Raw octal mode (e.g. 0o755).
    pub mode: u32,
}

impl PosixPermissions {
    /// Standard secure web directory permissions (0o755: rwxr-xr-x).
    pub const SECURE_DIR: Self = Self { mode: 0o755 };
    /// Standard secure file permissions (0o644: rw-r--r--).
    pub const SECURE_FILE: Self = Self { mode: 0o644 };
    /// Restrictive private file permissions (0o600: rw-------).
    pub const PRIVATE_FILE: Self = Self { mode: 0o600 };

    /// Creates permissions from raw octal mode.
    pub const fn from_octal(mode: u32) -> Self {
        Self { mode: mode & 0o777 }
    }

    /// True if Owner has Read permission.
    pub const fn owner_read(&self) -> bool {
        (self.mode & 0o400) != 0
    }
    /// True if Owner has Write permission.
    pub const fn owner_write(&self) -> bool {
        (self.mode & 0o200) != 0
    }
    /// True if Owner has Execute permission.
    pub const fn owner_exec(&self) -> bool {
        (self.mode & 0o100) != 0
    }

    /// True if Group has Read permission.
    pub const fn group_read(&self) -> bool {
        (self.mode & 0o040) != 0
    }
    /// True if Group has Write permission.
    pub const fn group_write(&self) -> bool {
        (self.mode & 0o020) != 0
    }
    /// True if Group has Execute permission.
    pub const fn group_exec(&self) -> bool {
        (self.mode & 0o010) != 0
    }

    /// True if Others have Read permission.
    pub const fn others_read(&self) -> bool {
        (self.mode & 0o004) != 0
    }
    /// True if Others have Write permission (dangerous for web roots!).
    pub const fn others_write(&self) -> bool {
        (self.mode & 0o002) != 0
    }
    /// True if Others have Execute permission.
    pub const fn others_exec(&self) -> bool {
        (self.mode & 0o001) != 0
    }

    /// Verifies that others do NOT have write permission on this web asset.
    pub fn verify_web_security(&self) -> ZeroResult<()> {
        if self.others_write() {
            // World-writable web files are an immediate security violation
            return Err(ZeroError::InvalidArgument);
        }
        Ok(())
    }

    /// Formats POSIX permissions into standard 9-character string `rwxrwxrwx`.
    pub fn to_display_string(&self) -> [u8; 9] {
        let mut out = [b'-'; 9];
        if self.owner_read() {
            out[0] = b'r';
        }
        if self.owner_write() {
            out[1] = b'w';
        }
        if self.owner_exec() {
            out[2] = b'x';
        }

        if self.group_read() {
            out[3] = b'r';
        }
        if self.group_write() {
            out[4] = b'w';
        }
        if self.group_exec() {
            out[5] = b'x';
        }

        if self.others_read() {
            out[6] = b'r';
        }
        if self.others_write() {
            out[7] = b'w';
        }
        if self.others_exec() {
            out[8] = b'x';
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posix_permissions_security() {
        let secure = PosixPermissions::from_octal(0o755);
        assert!(secure.verify_web_security().is_ok());
        let s = secure.to_display_string();
        assert_eq!(&s, b"rwxr-xr-x");

        let insecure = PosixPermissions::from_octal(0o777); // World-writable
        assert!(insecure.verify_web_security().is_err());
    }
}
