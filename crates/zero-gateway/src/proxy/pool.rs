//! # Upstream TCP Keep-Alive Connection Pool
//!
//! Maintains a pool of warm TCP streams connected to upstream servers to eliminate
//! connection setup overhead and TCP 3-way handshakes under extreme throughput.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

/// Manages reusable TCP streams connected to upstream destinations.
#[derive(Clone)]
pub struct UpstreamConnectionPool {
    pool: Arc<Mutex<HashMap<String, VecDeque<TcpStream>>>>,
    max_idle_per_host: usize,
}

impl UpstreamConnectionPool {
    /// Creates a new connection pool with a maximum idle stream count per host.
    pub fn new(max_idle_per_host: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(HashMap::new())),
            max_idle_per_host: max_idle_per_host.max(16),
        }
    }

    /// Acquires an idle TCP stream for the specified host:port destination.
    pub async fn acquire(&self, host_port: &str) -> Option<TcpStream> {
        let mut lock = self.pool.lock().await;
        if let Some(queue) = lock.get_mut(host_port) {
            if let Some(stream) = queue.pop_back() {
                return Some(stream);
            }
        }
        None
    }

    /// Returns a healthy TCP stream back to the pool for subsequent requests.
    pub async fn release(&self, host_port: String, stream: TcpStream) {
        let mut lock = self.pool.lock().await;
        let queue = lock.entry(host_port).or_insert_with(VecDeque::new);
        if queue.len() < self.max_idle_per_host {
            queue.push_back(stream);
        }
        // If queue exceeds capacity, stream is dropped and closed gracefully
    }
}
