//! Caddy v2 configuration generators for static websites, SPAs, and PHP-FPM fastcgi.

use crate::config::{VhostDescriptor, VhostKind};
use zero_core::error::{ZeroError, ZeroResult};

/// Builder and serializer for Caddy configuration blocks.
pub struct VhostBuilder<'a> {
    desc: VhostDescriptor<'a>,
}

impl<'a> VhostBuilder<'a> {
    /// Creates a new VhostBuilder for the specified domain.
    pub const fn new(domain: &'a str) -> Self {
        Self {
            desc: VhostDescriptor {
                domain,
                root_path: "/var/www/html",
                kind: VhostKind::Static,
                php_socket: None,
                enable_https: true,
                enable_compression: true,
            },
        }
    }

    /// Sets the document root path.
    pub const fn root(mut self, root_path: &'a str) -> Self {
        self.desc.root_path = root_path;
        self
    }

    /// Configures as a Single Page Application (fallback to /index.html).
    pub const fn spa(mut self) -> Self {
        self.desc.kind = VhostKind::SinglePageApp;
        self
    }

    /// Configures as a PHP application forwarding to a Unix domain socket.
    pub const fn php(mut self, socket_path: &'a str) -> Self {
        self.desc.kind = VhostKind::Php;
        self.desc.php_socket = Some(socket_path);
        self
    }

    /// Toggles Automatic HTTPS.
    pub const fn https(mut self, enable: bool) -> Self {
        self.desc.enable_https = enable;
        self
    }

    /// Returns the underlying descriptor.
    pub const fn build(&self) -> VhostDescriptor<'a> {
        self.desc
    }

    /// Generates a clean Caddyfile server block into `out_buf` without heap allocations.
    ///
    /// Returns total bytes written.
    pub fn write_caddyfile(&self, out_buf: &mut [u8]) -> ZeroResult<usize> {
        let mut cursor = 0;

        macro_rules! write_str {
            ($s:expr) => {
                let bytes = $s.as_bytes();
                if cursor + bytes.len() > out_buf.len() {
                    return Err(ZeroError::BufferOverflow);
                }
                out_buf[cursor..cursor + bytes.len()].copy_from_slice(bytes);
                cursor += bytes.len();
            };
        }

        // Domain header: e.g. "example.com" or "http://example.com" if https is disabled
        if !self.desc.enable_https {
            write_str!("http://");
        }
        write_str!(self.desc.domain);
        write_str!(" {\n    root * ");
        write_str!(self.desc.root_path);
        write_str!("\n");

        if self.desc.enable_compression {
            write_str!("    encode zstd gzip\n");
        }

        match self.desc.kind {
            VhostKind::Static => {
                write_str!("    file_server\n");
            }
            VhostKind::SinglePageApp => {
                write_str!("    try_files {path} /index.html\n    file_server\n");
            }
            VhostKind::Php => {
                if let Some(sock) = self.desc.php_socket {
                    write_str!("    php_fastcgi ");
                    write_str!(sock);
                    write_str!("\n    file_server\n");
                } else {
                    return Err(ZeroError::InvalidArgument);
                }
            }
        }

        write_str!("}\n");

        Ok(cursor)
    }

    /// Generates a Caddy v2 JSON Route representation for the Caddy REST API.
    pub fn write_json_route(&self, out_buf: &mut [u8]) -> ZeroResult<usize> {
        let mut cursor = 0;

        macro_rules! write_str {
            ($s:expr) => {
                let bytes = $s.as_bytes();
                if cursor + bytes.len() > out_buf.len() {
                    return Err(ZeroError::BufferOverflow);
                }
                out_buf[cursor..cursor + bytes.len()].copy_from_slice(bytes);
                cursor += bytes.len();
            };
        }

        write_str!("{\"match\":[{\"host\":[\"");
        write_str!(self.desc.domain);
        write_str!("\"]}],\"handle\":[{\"handler\":\"vars\",\"root\":\"");
        write_str!(self.desc.root_path);
        write_str!("\"},");

        if self.desc.kind == VhostKind::SinglePageApp {
            write_str!("{\"handler\":\"rewrite\",\"uri\":\"/index.html\"},");
        }

        if self.desc.kind == VhostKind::Php {
            if let Some(sock) = self.desc.php_socket {
                write_str!("{\"handler\":\"reverse_proxy\",\"transport\":{\"protocol\":\"fastcgi\",\"split_path\":[\".php\"]},\"upstreams\":[{\"dial\":\"");
                write_str!(sock);
                write_str!("\"}]},");
            } else {
                return Err(ZeroError::InvalidArgument);
            }
        }

        write_str!("{\"handler\":\"file_server\"}],\"terminal\":true}");

        Ok(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caddyfile_generation_static_and_php() {
        let mut buf = [0u8; 1024];

        // Static site
        let static_builder = VhostBuilder::new("static.io").root("/var/www/static");
        let len = static_builder.write_caddyfile(&mut buf).unwrap();
        let s = core::str::from_utf8(&buf[..len]).unwrap();
        assert!(s.contains("static.io {"));
        assert!(s.contains("root * /var/www/static"));
        assert!(s.contains("file_server"));

        // PHP site
        let php_builder = VhostBuilder::new("php.io")
            .root("/var/www/php")
            .php("unix//run/php/php8.2-fpm.sock");
        let len_php = php_builder.write_caddyfile(&mut buf).unwrap();
        let s_php = core::str::from_utf8(&buf[..len_php]).unwrap();
        assert!(s_php.contains("php_fastcgi unix//run/php/php8.2-fpm.sock"));
    }

    #[test]
    fn test_caddy_json_route() {
        let mut buf = [0u8; 1024];
        let builder = VhostBuilder::new("app.io").root("/var/www/app").spa();
        let len = builder.write_json_route(&mut buf).unwrap();
        let json_str = core::str::from_utf8(&buf[..len]).unwrap();
        assert!(json_str.contains("\"host\":[\"app.io\"]"));
        assert!(json_str.contains("\"root\":\"/var/www/app\""));
        assert!(json_str.contains("\"handler\":\"file_server\""));
    }
}
