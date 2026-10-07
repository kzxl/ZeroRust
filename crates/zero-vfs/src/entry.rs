use alloc::format;
use alloc::string::String;

/// File type in the virtual file system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsFileType {
    File,
    Directory,
    Symlink,
}

/// Metadata representation of a file or directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsEntry {
    /// Entry basename (e.g. "index.php").
    pub name: String,
    /// Relative path within the jail (e.g. "public/index.php").
    pub rel_path: String,
    /// Type of entry (File, Directory, Symlink).
    pub file_type: VfsFileType,
    /// File size in bytes (0 for directories).
    pub size_bytes: u64,
    /// Last modified timestamp in seconds since Unix epoch.
    pub modified_epoch_sec: u64,
    /// POSIX file permission mode (e.g. 0o755).
    pub posix_mode: u32,
}

impl VfsEntry {
    /// Creates a new `VfsEntry`.
    pub fn new(
        name: impl Into<String>,
        rel_path: impl Into<String>,
        file_type: VfsFileType,
        size_bytes: u64,
        modified_epoch_sec: u64,
        posix_mode: u32,
    ) -> Self {
        Self {
            name: name.into(),
            rel_path: rel_path.into(),
            file_type,
            size_bytes,
            modified_epoch_sec,
            posix_mode,
        }
    }

    /// Returns `true` if the entry is a regular file.
    pub fn is_file(&self) -> bool {
        matches!(self.file_type, VfsFileType::File)
    }

    /// Returns `true` if the entry is a directory.
    pub fn is_dir(&self) -> bool {
        matches!(self.file_type, VfsFileType::Directory)
    }

    /// Returns `true` if this file is hidden (starts with a dot).
    pub fn is_hidden(&self) -> bool {
        self.name.starts_with('.')
    }

    /// Returns the human-readable POSIX octal permission (e.g. "0755", "0644").
    pub fn format_posix_octal(&self) -> String {
        format!("{:04o}", self.posix_mode & 0o7777)
    }

    /// Formats file size into a human-readable string (B, KB, MB, GB).
    pub fn format_size(&self) -> String {
        if self.is_dir() {
            return String::from("-");
        }

        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;
        const GB: u64 = 1024 * MB;

        let b = self.size_bytes;
        if b < KB {
            format!("{b} B")
        } else if b < MB {
            let kb_int = b / KB;
            let kb_dec = ((b % KB) * 10) / KB;
            format!("{kb_int}.{kb_dec} KB")
        } else if b < GB {
            let mb_int = b / MB;
            let mb_dec = ((b % MB) * 10) / MB;
            format!("{mb_int}.{mb_dec} MB")
        } else {
            let gb_int = b / GB;
            let gb_dec = ((b % GB) * 10) / GB;
            format!("{gb_int}.{gb_dec} GB")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vfs_entry_formatting() {
        let entry = VfsEntry::new(
            "index.php",
            "public/index.php",
            VfsFileType::File,
            2500,
            1700000000,
            0o644,
        );

        assert!(entry.is_file());
        assert!(!entry.is_dir());
        assert!(!entry.is_hidden());
        assert_eq!(entry.format_posix_octal(), "0644");
        assert_eq!(entry.format_size(), "2.4 KB");

        let hidden_dir =
            VfsEntry::new(".git", ".git", VfsFileType::Directory, 0, 1700000000, 0o755);
        assert!(hidden_dir.is_dir());
        assert!(hidden_dir.is_hidden());
        assert_eq!(hidden_dir.format_posix_octal(), "0755");
        assert_eq!(hidden_dir.format_size(), "-");
    }

    #[test]
    fn test_size_formatting_scales() {
        let e1 = VfsEntry::new("f", "f", VfsFileType::File, 500, 0, 0o644);
        assert_eq!(e1.format_size(), "500 B");

        let e2 = VfsEntry::new("f", "f", VfsFileType::File, 1024 * 1024 * 5, 0, 0o644);
        assert_eq!(e2.format_size(), "5.0 MB");

        let e3 = VfsEntry::new(
            "f",
            "f",
            VfsFileType::File,
            1024 * 1024 * 1024 * 2 + 500 * 1024 * 1024,
            0,
            0o644,
        );
        assert_eq!(e3.format_size(), "2.4 GB");
    }
}
