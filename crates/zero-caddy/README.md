# zero-caddy

Pure Rust Caddy v2 configuration models, Caddyfile/JSON Vhost builders, and Administration REST API client for `ZPanl`.

## Features
- **Zero Heap Allocation**: Models and builders serialize Caddyfile and JSON configurations directly into memory buffers.
- **Static & SPA Vhost Builders**: Generates static file server routes and Single Page Application rewrite rules (`try_files {path} /index.html`).
- **PHP-FPM Reverse Proxy**: Generates FastCGI reverse proxy directives targeting Unix domain sockets (`php_fastcgi unix//run/php/php*.sock`).
- **Admin REST API**: Formats HTTP/1.1 requests targeting Caddy's dynamic administration socket (`http://localhost:2019/load`) for instant configuration reloading without process restart.
