//! Caddy v2 JSON Configuration models and route descriptors.

/// Website hosting category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VhostKind {
    /// Pure static website (HTML, CSS, JS, Images, WASM).
    Static,
    /// Single Page Application (React, Vue, Svelte) with fallback to `index.html`.
    SinglePageApp,
    /// Dynamic PHP application forwarded to PHP-FPM via Unix domain socket.
    Php,
}

/// Description of an individual VirtualHost managed by ZPanl in Caddy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VhostDescriptor<'a> {
    /// Domain name (e.g. "example.com" or "app.example.com").
    pub domain: &'a str,
    /// Absolute filesystem path to document root (e.g. "/var/www/example.com").
    pub root_path: &'a str,
    /// Vhost kind (Static, SPA, or PHP).
    pub kind: VhostKind,
    /// Unix domain socket for PHP-FPM (required if kind == VhostKind::Php).
    pub php_socket: Option<&'a str>,
    /// Enable Automatic HTTPS via Let's Encrypt / ZeroSSL.
    pub enable_https: bool,
    /// Enable Gzip / Zstandard response compression.
    pub enable_compression: bool,
}

impl<'a> Default for VhostDescriptor<'a> {
    fn default() -> Self {
        Self {
            domain: "localhost",
            root_path: "/var/www/html",
            kind: VhostKind::Static,
            php_socket: None,
            enable_https: true,
            enable_compression: true,
        }
    }
}
