//! # Sovereign Gateway Server Runtime & Connection Dispatcher
//!
//! Event-driven Tokio network listener dispatching connections into the Gravitational Pipeline.

use crate::auth::AuthManager;
use crate::cluster::ClusterRegistry;
use crate::config::GatewayConfig;
use crate::error::GatewayError;
use crate::pipeline::ratelimit::RateLimiterRegistry;
use crate::proxy::buffer_pool::BufferPool;
use crate::proxy::websocket::{is_websocket_upgrade, tunnel_bidirectional};
use crate::proxy::ReverseProxy;
use crate::routing::RouteRegistry;
use crate::telemetry::MetricsCollector;

use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Shared state container across all request handling tasks.
#[derive(Clone)]
pub struct GatewayState {
    pub routes: Arc<RouteRegistry>,
    pub clusters: Arc<ClusterRegistry>,
    pub rate_limiters: Arc<RateLimiterRegistry>,
    pub auth: Arc<AuthManager>,
    pub metrics: Arc<MetricsCollector>,
    pub proxy: Arc<ReverseProxy>,
}

/// Master ZGate Server coordinating routing, upstream clustering, and telemetry.
pub struct GatewayServer {
    config: GatewayConfig,
    state: Arc<GatewayState>,
}

impl GatewayServer {
    pub fn new(config: GatewayConfig) -> Self {
        let routes = Arc::new(RouteRegistry::new());
        routes.reload(config.routes.clone());

        let clusters = Arc::new(ClusterRegistry::new());
        clusters.reload(config.clusters.clone());

        let rate_limiters = Arc::new(RateLimiterRegistry::new());
        let auth = Arc::new(AuthManager::new(&config.server.jwt_secret));
        let metrics = Arc::new(MetricsCollector::new());
        let buffer_pool = BufferPool::new(128, 512);
        let proxy = Arc::new(ReverseProxy::new(
            buffer_pool,
            config.server.upstream_timeout_ms,
        ));

        let state = Arc::new(GatewayState {
            routes,
            clusters,
            rate_limiters,
            auth,
            metrics,
            proxy,
        });

        Self { config, state }
    }

    /// Accessors for runtime administration and dynamic hot-reloads
    pub fn routes(&self) -> &Arc<RouteRegistry> {
        &self.state.routes
    }

    pub fn clusters(&self) -> &Arc<ClusterRegistry> {
        &self.state.clusters
    }

    pub fn metrics(&self) -> &Arc<MetricsCollector> {
        &self.state.metrics
    }

    /// Starts both the data-plane proxy listener and the control-plane admin listener.
    pub async fn run(&self) -> Result<(), GatewayError> {
        let proxy_addr = format!("{}:{}", self.config.server.bind_address, self.config.server.port);
        let admin_addr = format!("{}:{}", self.config.server.bind_address, self.config.server.admin_port);

        let proxy_listener = TcpListener::bind(&proxy_addr).await.map_err(|e| GatewayError::IoFailure {
            details: format!("Failed to bind proxy listener on {proxy_addr}: {e}"),
        })?;

        let admin_listener = TcpListener::bind(&admin_addr).await.map_err(|e| GatewayError::IoFailure {
            details: format!("Failed to bind admin listener on {admin_addr}: {e}"),
        })?;

        // Spawn Admin Control-Plane Server
        let admin_metrics = Arc::clone(&self.state.metrics);
        let admin_clusters = Arc::clone(&self.state.clusters);
        tokio::spawn(async move {
            run_admin_server(admin_listener, admin_metrics, admin_clusters).await;
        });

        // Main Data-Plane Connection Accept Loop
        loop {
            match proxy_listener.accept().await {
                Ok((stream, addr)) => {
                    let client_ip = addr.ip().to_string();
                    let state = Arc::clone(&self.state);

                    tokio::spawn(async move {
                        state.metrics.conn_open();
                        handle_client(stream, client_ip, state.clone()).await;
                        state.metrics.conn_close();
                    });
                }
                Err(_) => {
                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                }
            }
        }
    }
}

async fn handle_client(mut stream: TcpStream, client_ip: String, state: Arc<GatewayState>) {
    let start_time = Instant::now();
    let mut header_buf = [0u8; 8192];
    let n = match stream.read(&mut header_buf).await {
        Ok(n) if n > 0 => n,
        _ => return,
    };

    let header_bytes = &header_buf[..n];
    let header_str = match std::str::from_utf8(header_bytes) {
        Ok(s) => s,
        Err(_) => {
            let _ = send_simple_response(&mut stream, 400, "Bad Request: Malformed UTF-8").await;
            state.metrics.record_request(400, start_time.elapsed().as_micros() as u64);
            return;
        }
    };

    let first_line = match header_str.lines().next() {
        Some(l) => l,
        None => return,
    };

    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        let _ = send_simple_response(&mut stream, 400, "Bad Request").await;
        state.metrics.record_request(400, start_time.elapsed().as_micros() as u64);
        return;
    }

    let path = parts[1];

    // Internal health probe shortcut
    if path == "/health" {
        let _ = send_json_response(&mut stream, 200, r#"{"status":"healthy","engine":"zgate-rust"}"#).await;
        state.metrics.record_request(200, start_time.elapsed().as_micros() as u64);
        return;
    }

    // 1. Prefix Trie Route Match (O(K))
    let route = match state.routes.match_route(path) {
        Some(r) => r,
        None => {
            let _ = send_simple_response(&mut stream, 404, "404 Not Found: No route matched").await;
            state.metrics.record_request(404, start_time.elapsed().as_micros() as u64);
            return;
        }
    };

    // 2. Token Bucket Rate Limiting
    if route.rate_limit_per_second > 0
        && !state.rate_limiters.check_allowed(&route.route_id, route.rate_limit_per_second)
    {
        let _ = send_simple_response(&mut stream, 429, "429 Too Many Requests").await;
        state.metrics.record_request(429, start_time.elapsed().as_micros() as u64);
        return;
    }

    // 3. JWT Authentication Verification
    if route.auth_required {
        let mut token = None;
        for line in header_str.lines() {
            if line.to_ascii_lowercase().starts_with("authorization:") {
                let val = line[14..].trim();
                token = Some(val);
                break;
            }
        }

        match token {
            Some(t) => {
                if let Err(e) = state.auth.verify_token(t) {
                    let _ = send_simple_response(&mut stream, 401, &format!("401 Unauthorized: {e}")).await;
                    state.metrics.record_request(401, start_time.elapsed().as_micros() as u64);
                    return;
                }
            }
            None => {
                let _ = send_simple_response(&mut stream, 401, "401 Unauthorized: Missing Token").await;
                state.metrics.record_request(401, start_time.elapsed().as_micros() as u64);
                return;
            }
        }
    }

    // 4. Cluster & Destination Selection
    let cluster = match state.clusters.get(&route.cluster_id) {
        Some(c) => c,
        None => {
            let _ = send_simple_response(&mut stream, 502, "502 Bad Gateway: Cluster not found").await;
            state.metrics.record_request(502, start_time.elapsed().as_micros() as u64);
            return;
        }
    };

    let destination = match cluster.select_destination() {
        Some(d) => d,
        None => {
            let _ = send_simple_response(&mut stream, 502, "502 Bad Gateway: No healthy upstream").await;
            state.metrics.record_request(502, start_time.elapsed().as_micros() as u64);
            return;
        }
    };

    // 5. WebSocket Tunneling vs L7 HTTP Forwarding
    let header_lower = header_str.to_ascii_lowercase();
    if is_websocket_upgrade(&header_lower) {
        let raw_addr = destination.address.trim_start_matches("http://").trim_start_matches("https://");
        if let Ok(upstream) = TcpStream::connect(raw_addr).await {
            let _ = tunnel_bidirectional(stream, upstream).await;
            state.metrics.record_request(101, start_time.elapsed().as_micros() as u64);
        }
    } else {
        match state.proxy.forward(&mut stream, &destination, &route, header_bytes, &client_ip).await {
            Ok(status) => {
                state.metrics.record_request(status, start_time.elapsed().as_micros() as u64);
            }
            Err(_) => {
                let _ = send_simple_response(&mut stream, 502, "502 Bad Gateway: Upstream failure").await;
                state.metrics.record_request(502, start_time.elapsed().as_micros() as u64);
            }
        }
    }
}

async fn run_admin_server(listener: TcpListener, metrics: Arc<MetricsCollector>, clusters: Arc<ClusterRegistry>) {
    loop {
        if let Ok((mut stream, _)) = listener.accept().await {
            let metrics = Arc::clone(&metrics);
            let clusters = Arc::clone(&clusters);
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                if let Ok(n) = stream.read(&mut buf).await {
                    let req_str = std::str::from_utf8(&buf[..n]).unwrap_or("");
                    if req_str.contains("/metrics") {
                        let prom = metrics.to_prometheus();
                        let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{prom}", prom.len());
                        let _ = stream.write_all(resp.as_bytes()).await;
                    } else if req_str.contains("/admin/stats") {
                        let snap = metrics.snapshot();
                        let json = serde_json::to_string(&snap).unwrap_or_default();
                        let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{json}", json.len());
                        let _ = stream.write_all(resp.as_bytes()).await;
                    } else {
                        let active_clusters = clusters.list_all();
                        let cluster_summary: Vec<_> = active_clusters.iter().map(|c| {
                            serde_json::json!({
                                "cluster_id": c.cluster_id,
                                "name": c.name,
                                "destinations": c.destinations.iter().map(|d| {
                                    serde_json::json!({
                                        "address": d.address,
                                        "healthy": d.is_healthy.load(std::sync::atomic::Ordering::Relaxed),
                                        "in_flight": d.in_flight.load(std::sync::atomic::Ordering::Relaxed),
                                    })
                                }).collect::<Vec<_>>()
                            })
                        }).collect();
                        let json = serde_json::json!({
                            "status": "online",
                            "engine": "ZGate Sovereign Rust Core",
                            "clusters": cluster_summary
                        }).to_string();
                        let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{json}", json.len());
                        let _ = stream.write_all(resp.as_bytes()).await;
                    }
                }
            });
        }
    }
}

async fn send_simple_response(stream: &mut TcpStream, code: u16, msg: &str) -> Result<(), std::io::Error> {
    let body = format!("{msg}\r\n");
    let resp = format!("HTTP/1.1 {code} Status\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    stream.write_all(resp.as_bytes()).await
}

async fn send_json_response(stream: &mut TcpStream, code: u16, json: &str) -> Result<(), std::io::Error> {
    let resp = format!("HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}", json.len());
    stream.write_all(resp.as_bytes()).await
}
