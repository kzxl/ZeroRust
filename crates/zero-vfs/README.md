# zero-vfs 🛡️📁

> **Sovereign Virtual File System and Security Sandbox for Web Panels & Embedded Edge Controllers.**  
> Part of the [`ZeroRust`](../../) ecosystem and security layer for **ZPanl**.

---

## 🌟 Overview

`zero-vfs` provides path sanitization, traversal attack prevention, metadata normalization, atomic staging protocols, and zero-allocation MIME resolution.

### Core Capabilities

1. **`JailSandbox`**:
   - Enforces containment within a defined directory boundary (e.g. `/var/www/site`).
   - Normalizes path separators and relative segments (`.`, `..`).
   - Rejects null bytes and traversal escapes (`../../etc/passwd`).
2. **`VfsEntry` & `VfsFileType`**:
   - High-level representation of directory listings.
   - Formats human-readable sizes (B, KB, MB, GB) and octal POSIX permissions (`0755`, `0644`).
3. **`detect_mime` & `is_text_editable`**:
   - Static string slice lookup for common web files without heap allocation.
   - Detects text files for in-browser editing.
4. **`AtomicPathBuilder`**:
   - Guarantees same-filesystem temporary and backup paths to ensure POSIX `rename(2)` atomicity.

---

## 🚀 Quick Example

```rust
use zero_vfs::{JailSandbox, detect_mime, is_text_editable};

// 1. Confine file access to site root
let jail = JailSandbox::new("/var/www/site1").unwrap();

// Safe relative resolution
let safe_path = jail.resolve("public/index.php").unwrap();
assert_eq!(safe_path, "/var/www/site1/public/index.php");

// Malicious traversal is rejected with SecurityViolation
assert!(jail.resolve("../../etc/passwd").is_err());

// 2. MIME & Editing detection
assert_eq!(detect_mime("index.php"), "application/x-httpd-php");
assert!(is_text_editable("index.php"));
```

---

## 📜 License

Licensed under the MIT License.
