//! # High-Throughput L7 HTTP/1.1 & WebSocket Reverse Proxy Engine
//!
//! Sub-millisecond stream forwarding utilizing aligned buffer pools and connection multiplexing.

pub mod buffer_pool;
pub mod websocket;

use buffer_pool::BufferPool;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{timeout, Duration};

use crate::cluster::destination::Destination;
use crate::error::GatewayError;
use crate::routing::rule::RouteRule;

/// Reverse proxy engine for streaming L7 requests to upstreams.
pub struct ReverseProxy {
    buffer_pool: Arc<BufferPool>,
    timeout_duration: Duration,
}

impl ReverseProxy {
    pub fn new(buffer_pool: Arc<BufferPool>, timeout_ms: u64) -> Self {
        Self {
            buffer_pool,
            timeout_duration: Duration::from_millis(timeout_ms.max(1000)),
        }
    }

    /// Forward client request to selected destination and stream response back.
    pub async fn forward(
        &self,
        client_stream: &mut TcpStream,
        destination: &Arc<Destination>,
        route: &Arc<RouteRule>,
        raw_header_bytes: &[u8],
        client_ip: &str,
    ) -> Result<u16, GatewayError> {
        let start_time = Instant::now();
        destination.record_start();

        let raw_addr = destination.address.trim();
        let host_port = raw_addr
            .trim_start_matches("http://")
            .trim_start_matches("https://");

        // Connect to upstream with timeout
        let upstream_conn = timeout(Duration::from_millis(3000), TcpStream::connect(host_port)).await;
        let mut upstream = match upstream_conn {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => {
                destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                return Err(GatewayError::UpstreamConnectionFailed {
                    address: destination.address.clone(),
                    details: e.to_string(),
                });
            }
            Err(_) => {
                destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                return Err(GatewayError::UpstreamTimeout { timeout_ms: 3000 });
            }
        };

        // Prepare rewritten request
        let header_str = match std::str::from_utf8(raw_header_bytes) {
            Ok(s) => s,
            Err(_) => {
                destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                return Err(GatewayError::MalformedRequest {
                    reason: "Header is not valid UTF-8".to_string(),
                });
            }
        };

        let mut lines = header_str.lines();
        let request_line = match lines.next() {
            Some(l) => l,
            None => {
                destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                return Err(GatewayError::MalformedRequest {
                    reason: "Empty request line".to_string(),
                });
            }
        };

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 3 {
            destination.record_finish(false, start_time.elapsed().as_micros() as u64);
            return Err(GatewayError::MalformedRequest {
                reason: "Malformed request line".to_string(),
            });
        }

        let method = parts[0];
        let original_path = parts[1];
        let version = parts[2];

        let rewritten_path = route.rewrite_path(original_path);
        let new_req_line = format!("{method} {rewritten_path} {version}\r\n");

        // Reconstruct modified request headers with gateway metadata
        let mut forward_headers = String::with_capacity(raw_header_bytes.len() + 256);
        forward_headers.push_str(&new_req_line);

        for line in lines {
            if line.is_empty() {
                break;
            }
            // Strip client hop-by-hop Host or override
            if line.to_ascii_lowercase().starts_with("host:") {
                forward_headers.push_str(&format!("Host: {host_port}\r\n"));
            } else {
                forward_headers.push_str(line);
                forward_headers.push_str("\r\n");
            }
        }

        forward_headers.push_str(&format!("X-Forwarded-For: {client_ip}\r\n"));
        forward_headers.push_str(&format!("X-Gateway-RouteId: {}\r\n", route.route_id));
        forward_headers.push_str(&format!("X-Gateway-ClusterId: {}\r\n\r\n", route.cluster_id));

        // Send rewritten headers to upstream
        if let Err(e) = upstream.write_all(forward_headers.as_bytes()).await {
            destination.record_finish(false, start_time.elapsed().as_micros() as u64);
            return Err(GatewayError::IoFailure {
                details: format!("Failed to write headers to upstream: {e}"),
            });
        }

        // Stream response from upstream back to client using pool buffer
        let mut buffer = self.buffer_pool.acquire();
        let mut status_code: u16 = 200;
        let mut first_chunk = true;

        loop {
            let read_res = timeout(self.timeout_duration, upstream.read(&mut buffer[..])).await;
            match read_res {
                Ok(Ok(0)) => break, // EOF
                Ok(Ok(n)) => {
                    if first_chunk {
                        first_chunk = false;
                        if let Ok(first_str) = std::str::from_utf8(&buffer[..n.min(64)]) {
                            if let Some(status_str) = first_str.split_whitespace().nth(1) {
                                status_code = status_str.parse::<u16>().unwrap_or(200);
                            }
                        }
                    }
                    if let Err(e) = client_stream.write_all(&buffer[..n]).await {
                        destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                        return Err(GatewayError::IoFailure {
                            details: format!("Failed to write to client: {e}"),
                        });
                    }
                }
                Ok(Err(e)) => {
                    destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                    return Err(GatewayError::IoFailure {
                        details: format!("Upstream read error: {e}"),
                    });
                }
                Err(_) => {
                    destination.record_finish(false, start_time.elapsed().as_micros() as u64);
                    return Err(GatewayError::UpstreamTimeout {
                        timeout_ms: self.timeout_duration.as_millis() as u64,
                    });
                }
            }
        }

        destination.record_finish(status_code < 500, start_time.elapsed().as_micros() as u64);
        Ok(status_code)
    }
}
