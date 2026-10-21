//! # Cluster Management & Upstream Orchestration
//!
//! Cluster registry supporting dynamic addition, removal, and health orchestration.

pub mod balancer;
pub mod destination;
pub mod health;

use balancer::{LoadBalancer, LoadBalancingPolicy};
use destination::Destination;
use health::HealthProber;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use crate::config::ClusterConfig;

/// A configured cluster containing target destinations and a load balancer.
pub struct Cluster {
    pub cluster_id: String,
    pub name: String,
    pub destinations: Vec<Arc<Destination>>,
    pub balancer: LoadBalancer,
}

impl Cluster {
    pub fn new(config: ClusterConfig) -> Self {
        let policy = LoadBalancingPolicy::from(config.load_balancing_policy.as_str());
        let destinations: Vec<Arc<Destination>> = config
            .destinations
            .into_iter()
            .map(|d| Arc::new(Destination::from(d)))
            .collect();

        // Spawn active health checker if interval > 0
        if config.health_check_interval_sec > 0 && !destinations.is_empty() {
            HealthProber::spawn_checker(
                destinations.clone(),
                config.health_check_path,
                config.health_check_interval_sec,
            );
        }

        Self {
            cluster_id: config.cluster_id,
            name: config.name,
            destinations,
            balancer: LoadBalancer::new(policy),
        }
    }

    /// Selects an optimal upstream destination using the configured policy.
    #[inline]
    pub fn select_destination(&self) -> Option<Arc<Destination>> {
        self.balancer.select(&self.destinations).cloned()
    }
}

/// Thread-safe Cluster Registry for lookups and dynamic reloads.
#[derive(Clone)]
pub struct ClusterRegistry {
    clusters: Arc<RwLock<HashMap<String, Arc<Cluster>>>>,
}

impl Default for ClusterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterRegistry {
    pub fn new() -> Self {
        Self {
            clusters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Retrieves a cluster by its unique cluster_id.
    #[inline]
    pub fn get(&self, cluster_id: &str) -> Option<Arc<Cluster>> {
        let guard = self.clusters.read();
        guard.get(cluster_id).cloned()
    }

    /// Atomically replaces all clusters with a new configuration set.
    pub fn reload(&self, configs: Vec<ClusterConfig>) {
        let mut map = HashMap::new();
        for conf in configs {
            if conf.is_active {
                map.insert(conf.cluster_id.clone(), Arc::new(Cluster::new(conf)));
            }
        }
        let mut guard = self.clusters.write();
        *guard = map;
    }

    /// Returns a snapshot of all active clusters for admin/telemetry endpoints.
    pub fn list_all(&self) -> Vec<Arc<Cluster>> {
        let guard = self.clusters.read();
        guard.values().cloned().collect()
    }
}
