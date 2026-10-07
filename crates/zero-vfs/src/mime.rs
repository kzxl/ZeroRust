/// Determines the MIME content type from a filename or path without dynamic allocation.
pub fn detect_mime(filename: &str) -> &'static str {
    let ext = match filename.rfind('.') {
        Some(idx) => &filename[idx + 1..],
        None => return "application/octet-stream",
    };

    // Case-insensitive match on common web extensions
    if ext.eq_ignore_ascii_case("html") || ext.eq_ignore_ascii_case("htm") {
        "text/html; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("php") {
        "application/x-httpd-php"
    } else if ext.eq_ignore_ascii_case("css") {
        "text/css; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("js") || ext.eq_ignore_ascii_case("mjs") {
        "application/javascript; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("json") {
        "application/json"
    } else if ext.eq_ignore_ascii_case("xml") {
        "application/xml"
    } else if ext.eq_ignore_ascii_case("txt")
        || ext.eq_ignore_ascii_case("log")
        || ext.eq_ignore_ascii_case("md")
        || ext.eq_ignore_ascii_case("ini")
        || ext.eq_ignore_ascii_case("conf")
        || ext.eq_ignore_ascii_case("env")
        || ext.eq_ignore_ascii_case("yml")
        || ext.eq_ignore_ascii_case("yaml")
    {
        "text/plain; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("png") {
        "image/png"
    } else if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
        "image/jpeg"
    } else if ext.eq_ignore_ascii_case("gif") {
        "image/gif"
    } else if ext.eq_ignore_ascii_case("svg") {
        "image/svg+xml"
    } else if ext.eq_ignore_ascii_case("webp") {
        "image/webp"
    } else if ext.eq_ignore_ascii_case("ico") {
        "image/x-icon"
    } else if ext.eq_ignore_ascii_case("woff2") {
        "font/woff2"
    } else if ext.eq_ignore_ascii_case("woff") {
        "font/woff"
    } else if ext.eq_ignore_ascii_case("ttf") {
        "font/ttf"
    } else if ext.eq_ignore_ascii_case("zip") {
        "application/zip"
    } else if ext.eq_ignore_ascii_case("tar") {
        "application/x-tar"
    } else if ext.eq_ignore_ascii_case("gz") {
        "application/gzip"
    } else if ext.eq_ignore_ascii_case("pdf") {
        "application/pdf"
    } else {
        "application/octet-stream"
    }
}

/// Returns `true` if the file can be safely viewed and edited in a text editor.
pub fn is_text_editable(filename: &str) -> bool {
    let mime = detect_mime(filename);
    mime.starts_with("text/")
        || mime == "application/json"
        || mime == "application/javascript; charset=utf-8"
        || mime == "application/xml"
        || mime == "application/x-httpd-php"
}

/// Returns `true` if the file is an archive capable of extraction or creation.
pub fn is_archive(filename: &str) -> bool {
    let ext = match filename.rfind('.') {
        Some(idx) => &filename[idx + 1..],
        None => return false,
    };
    ext.eq_ignore_ascii_case("zip")
        || ext.eq_ignore_ascii_case("tar")
        || ext.eq_ignore_ascii_case("gz")
        || ext.eq_ignore_ascii_case("tgz")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_detection() {
        assert_eq!(detect_mime("index.php"), "application/x-httpd-php");
        assert_eq!(detect_mime("style.CSS"), "text/css; charset=utf-8");
        assert_eq!(
            detect_mime("app.bundle.JS"),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(detect_mime("config.json"), "application/json");
        assert_eq!(detect_mime("logo.svg"), "image/svg+xml");
        assert_eq!(detect_mime("archive.zip"), "application/zip");
        assert_eq!(detect_mime("binary.unknown"), "application/octet-stream");
        assert_eq!(detect_mime("no_extension"), "application/octet-stream");
    }

    #[test]
    fn test_is_text_editable() {
        assert!(is_text_editable("index.php"));
        assert!(is_text_editable("nginx.conf"));
        assert!(is_text_editable(".env"));
        assert!(is_text_editable("README.md"));
        assert!(!is_text_editable("image.png"));
        assert!(!is_text_editable("package.zip"));
    }

    #[test]
    fn test_is_archive() {
        assert!(is_archive("backup.zip"));
        assert!(is_archive("source.tar"));
        assert!(is_archive("build.tar.gz"));
        assert!(!is_archive("file.txt"));
    }
}
