//! PHP-FPM worker pool configuration generator.

use zero_core::error::{ZeroError, ZeroResult};

/// PHP-FPM process manager mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessManagerType {
    /// On-demand: forks child processes only when connections arrive (ideal for low-memory VPS).
    OnDemand,
    /// Dynamic: keeps minimum spare workers and scales up to max_children.
    Dynamic,
    /// Static: fixed number of workers running continuously.
    Static,
}

impl ProcessManagerType {
    /// Returns the configuration string literal.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::OnDemand => "ondemand",
            Self::Dynamic => "dynamic",
            Self::Static => "static",
        }
    }
}

/// Configuration settings for an isolated PHP-FPM worker pool (`pool.d/site.conf`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhpPoolConfig<'a> {
    /// Unique pool identifier name (e.g. "site1").
    pub pool_name: &'a str,
    /// OS username under which the pool processes execute.
    pub user: &'a str,
    /// OS group name.
    pub group: &'a str,
    /// Absolute path to the Unix domain socket (e.g. "/run/php/php8.2-fpm-site1.sock").
    pub listen_socket: &'a str,
    /// Process manager scheduling mode.
    pub pm: ProcessManagerType,
    /// Maximum simultaneous child worker processes.
    pub max_children: u32,
    /// Idle process timeout in seconds (for ondemand mode).
    pub process_idle_timeout_sec: u32,
    /// Maximum memory limit in Megabytes (e.g. 128, 256).
    pub memory_limit_mb: u32,
    /// Maximum execution time in seconds (e.g. 30, 60).
    pub max_execution_time_sec: u32,
}

impl<'a> Default for PhpPoolConfig<'a> {
    fn default() -> Self {
        Self {
            pool_name: "www",
            user: "www-data",
            group: "www-data",
            listen_socket: "/run/php/php-fpm.sock",
            pm: ProcessManagerType::OnDemand,
            max_children: 5,
            process_idle_timeout_sec: 10,
            memory_limit_mb: 128,
            max_execution_time_sec: 30,
        }
    }
}

impl<'a> PhpPoolConfig<'a> {
    /// Generates PHP-FPM INI configuration text into `out_buf` without heap allocations.
    ///
    /// Returns total bytes written.
    pub fn write_ini_config(&self, out_buf: &mut [u8]) -> ZeroResult<usize> {
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

        macro_rules! write_u32 {
            ($val:expr) => {
                let mut tmp = [0u8; 10];
                let mut v = $val;
                let mut idx = 10;
                if v == 0 {
                    idx -= 1;
                    tmp[idx] = b'0';
                } else {
                    while v > 0 {
                        idx -= 1;
                        tmp[idx] = b'0' + (v % 10) as u8;
                        v /= 10;
                    }
                }
                write_str!(core::str::from_utf8(&tmp[idx..]).unwrap());
            };
        }

        write_str!("[");
        write_str!(self.pool_name);
        write_str!("]\nuser = ");
        write_str!(self.user);
        write_str!("\ngroup = ");
        write_str!(self.group);
        write_str!("\nlisten = ");
        write_str!(self.listen_socket);
        write_str!("\nlisten.owner = ");
        write_str!(self.user);
        write_str!("\nlisten.group = ");
        write_str!(self.group);
        write_str!("\nlisten.mode = 0660\n");

        write_str!("pm = ");
        write_str!(self.pm.as_str());
        write_str!("\npm.max_children = ");
        write_u32!(self.max_children);
        write_str!("\npm.process_idle_timeout = ");
        write_u32!(self.process_idle_timeout_sec);
        write_str!("s\n");

        write_str!("php_admin_value[memory_limit] = ");
        write_u32!(self.memory_limit_mb);
        write_str!("M\nphp_admin_value[max_execution_time] = ");
        write_u32!(self.max_execution_time_sec);
        write_str!("\n");

        Ok(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_generation() {
        let config = PhpPoolConfig {
            pool_name: "site_alpha",
            user: "vhost_alpha",
            group: "vhost_alpha",
            listen_socket: "/run/php/site_alpha.sock",
            pm: ProcessManagerType::OnDemand,
            max_children: 10,
            process_idle_timeout_sec: 15,
            memory_limit_mb: 256,
            max_execution_time_sec: 60,
        };

        let mut buf = [0u8; 1024];
        let len = config.write_ini_config(&mut buf).unwrap();
        let s = core::str::from_utf8(&buf[..len]).unwrap();

        assert!(s.contains("[site_alpha]"));
        assert!(s.contains("user = vhost_alpha"));
        assert!(s.contains("pm = ondemand"));
        assert!(s.contains("pm.max_children = 10"));
        assert!(s.contains("php_admin_value[memory_limit] = 256M"));
    }
}
