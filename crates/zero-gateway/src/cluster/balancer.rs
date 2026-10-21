//! # Atomic Multi-Strategy Load Balancing Engine
//!
//! Lock-free round-robin, least-connections, and weighted selection algorithms.

use super::destination::Destination;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Supported load balancing distribution strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingPolicy {
    RoundRobin,
    LeastConnections,
    Random,
}

impl From<&str> for LoadBalancingPolicy {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "leastconnections" | "least_connections" | "leastconn" => Self::LeastConnections,
            "random" => Self::Random,
            _ => Self::RoundRobin,
        }
    }
}

/// State for selecting upstream destinations within a cluster.
#[derive(Debug)]
pub struct LoadBalancer {
    policy: LoadBalancingPolicy,
    counter: AtomicUsize,
}

impl LoadBalancer {
    pub fn new(policy: LoadBalancingPolicy) -> Self {
        Self {
            policy,
            counter: AtomicUsize::new(0),
        }
    }

    /// Selects an optimal destination from the candidate list.
    pub fn select<'a>(&self, destinations: &'a [Arc<Destination>]) -> Option<&'a Arc<Destination>> {
        if destinations.is_empty() {
            return None;
        }

        // Fast-path: Filter available healthy nodes
        let available: Vec<&'a Arc<Destination>> = destinations
            .iter()
            .filter(|d| d.is_available())
            .collect();

        // If no healthy nodes are found, gracefully degrade to any active node
        let candidates = if available.is_empty() {
            let active: Vec<&'a Arc<Destination>> = destinations.iter().filter(|d| d.is_active).collect();
            if active.is_empty() {
                return None;
            }
            active
        } else {
            available
        };

        match self.policy {
            LoadBalancingPolicy::RoundRobin => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
                candidates.get(idx).copied()
            }
            LoadBalancingPolicy::LeastConnections => {
                let mut best_node = candidates.first().copied();
                let mut min_conns = usize::MAX;

                for node in &candidates {
                    let conns = node.in_flight.load(Ordering::Relaxed);
                    if conns < min_conns {
                        min_conns = conns;
                        best_node = Some(node);
                    }
                }
                best_node
            }
            LoadBalancingPolicy::Random => {
                // Use counter increment mixed with bitshift for lock-free pseudorandom index
                let val = self.counter.fetch_add(1, Ordering::Relaxed);
                let mixed = val.wrapping_mul(0x517cc1b727220a95);
                let idx = mixed % candidates.len();
                candidates.get(idx).copied()
            }
        }
    }
}
