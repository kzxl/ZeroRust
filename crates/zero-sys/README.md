# zero-sys

Native Linux system metrics (`/proc` zero-allocation parser), Systemd service management, and POSIX permissions for `ZPanl` in `ZeroRust`.

## Features
- **Zero-Allocation `/proc` Telemetry**: Parses `/proc/stat` (CPU usage %), `/proc/meminfo` (RAM usage & buffers), and `/proc/net/dev` (network RX/TX) without allocating memory.
- **Systemd Control**: Structured `systemctl` command generator and status parser for services (`caddy`, `php*-fpm`, `mariadb`).
- **POSIX Web Root Security**: Octal file permission validator enforcing secure non-world-writable web roots.
