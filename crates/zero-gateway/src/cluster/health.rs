//! # Active Upstream Health Prober & Circuit Breaker
//!
//! Autonomous Tokio background task verifying upstream availability.

use super::destination::Destination;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::sleep;

/// Health probe coordinator for a cluster's destinations.
pub struct HealthProber;

impl HealthProber {
    /// Spawns a background probing loop for the given list of destinations.
    pub fn spawn_checker(
        destinations: Vec<Arc<Destination>>,
        _check_path: String,
        interval_sec: u64,
    ) {
        let interval = Duration::from_secs(interval_sec.max(2));

        tokio::spawn(async move {
            loop {
                sleep(interval).await;

                for dest in &destinations {
                    if !dest.is_active {
                        continue;
                    }

                    let raw_addr = dest.address.trim();
                    let host_port = raw_addr
                        .trim_start_matches("http://")
                        .trim_start_matches("https://");

                    let is_alive = matches!(
                        tokio::time::timeout(
                            Duration::from_millis(1500),
                            TcpStream::connect(host_port),
                        )
                        .await,
                        Ok(Ok(_))
                    );

                    dest.mark_health(is_alive);
                }
            }
        });
    }
}
