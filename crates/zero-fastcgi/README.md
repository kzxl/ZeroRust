# zero-fastcgi

Pure Rust FastCGI v1.0 binary protocol framing, PHP-FPM Unix socket communication, and worker pool configuration generator for `ZPanl`.

## Features
- **Zero-Allocation FastCGI Protocol**: Record serializer and deserializer for `FCGI_BEGIN_REQUEST`, `FCGI_PARAMS`, `FCGI_STDIN`, `FCGI_STDOUT`, and `FCGI_END_REQUEST`.
- **FastCGI Request Builder**: Assembles standard CGI environment headers (`SCRIPT_FILENAME`, `DOCUMENT_ROOT`, `REQUEST_METHOD`) into a single network buffer.
- **PHP-FPM Pool Configuration**: Generates isolated `pool.d/*.conf` configurations enforcing per-site memory limits, timeout parameters, and on-demand process manager settings.
