//! Caddy Admin REST API request framing and client helpers.

use zero_core::error::{ZeroError, ZeroResult};

/// Helper to build HTTP requests targeted at the Caddy Administration endpoint (`localhost:2019`).
pub struct CaddyApiEndpoint;

impl CaddyApiEndpoint {
    /// Default Caddy administration API host and port.
    pub const DEFAULT_HOST: &'static str = "localhost";
    /// Default Caddy administration port.
    pub const DEFAULT_PORT: u16 = 2019;

    /// Formats an HTTP/1.1 POST `/load` request to inject a JSON or Caddyfile configuration into Caddy.
    pub fn build_load_request(
        payload: &[u8],
        is_caddyfile: bool,
        out_buf: &mut [u8],
    ) -> ZeroResult<usize> {
        let content_type = if is_caddyfile {
            "text/caddyfile"
        } else {
            "application/json"
        };

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

        write_str!("POST /load HTTP/1.1\r\nHost: localhost:2019\r\nContent-Type: ");
        write_str!(content_type);
        write_str!("\r\nContent-Length: ");
        write_u32!(payload.len() as u32);
        write_str!("\r\nConnection: close\r\n\r\n");

        if cursor + payload.len() > out_buf.len() {
            return Err(ZeroError::BufferOverflow);
        }
        out_buf[cursor..cursor + payload.len()].copy_from_slice(payload);
        cursor += payload.len();

        Ok(cursor)
    }

    /// Formats an HTTP/1.1 GET `/config/` request to fetch active running configuration from Caddy.
    pub fn build_get_config_request(out_buf: &mut [u8]) -> ZeroResult<usize> {
        let req = b"GET /config/ HTTP/1.1\r\nHost: localhost:2019\r\nConnection: close\r\n\r\n";
        if out_buf.len() < req.len() {
            return Err(ZeroError::BufferOverflow);
        }
        out_buf[..req.len()].copy_from_slice(req);
        Ok(req.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_request_builder() {
        let payload = b"{\"apps\":{}}";
        let mut buf = [0u8; 256];
        let len = CaddyApiEndpoint::build_load_request(payload, false, &mut buf).unwrap();
        let s = core::str::from_utf8(&buf[..len]).unwrap();

        assert!(s.starts_with("POST /load HTTP/1.1"));
        assert!(s.contains("Content-Type: application/json"));
        assert!(s.contains("Content-Length: 11"));
        assert!(s.ends_with("{\"apps\":{}}"));
    }
}
