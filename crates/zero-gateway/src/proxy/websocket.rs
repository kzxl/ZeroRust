//! # Full-Duplex WebSocket Tunneling Engine
//!
//! Detects WebSocket handshake headers and establishes bidirectional zero-copy TCP stream tunneling.

use std::io::Result as IoResult;
use tokio::io::copy_bidirectional;
use tokio::net::TcpStream;

/// Checks if an HTTP header block indicates a WebSocket upgrade request.
#[inline]
pub fn is_websocket_upgrade(headers_lower: &str) -> bool {
    headers_lower.contains("upgrade: websocket")
        || (headers_lower.contains("upgrade:") && headers_lower.contains("websocket"))
}

/// Tunnels raw bidirectional TCP frames between client and upstream destination.
pub async fn tunnel_bidirectional(
    mut client_stream: TcpStream,
    mut upstream_stream: TcpStream,
) -> IoResult<(u64, u64)> {
    copy_bidirectional(&mut client_stream, &mut upstream_stream).await
}
