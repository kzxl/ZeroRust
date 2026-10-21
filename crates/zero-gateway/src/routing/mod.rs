//! # Dynamic Routing Subsystem
//!
//! Route registry supporting lock-free concurrent reads and atomic hot-reloading.

pub mod rule;
pub mod trie;

use parking_lot::RwLock;
use rule::RouteRule;
use std::sync::Arc;
use trie::RouteTrie;
use crate::config::RouteConfig;

/// Thread-safe Route Registry for atomic path lookups and runtime hot-reloads.
#[derive(Clone)]
pub struct RouteRegistry {
    trie: Arc<RwLock<RouteTrie>>,
}

impl Default for RouteRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteRegistry {
    pub fn new() -> Self {
        Self {
            trie: Arc::new(RwLock::new(RouteTrie::new())),
        }
    }

    /// Match an incoming request path against the compiled Radix Trie.
    #[inline]
    pub fn match_route(&self, path: &str) -> Option<Arc<RouteRule>> {
        let trie = self.trie.read();
        trie.match_path(path)
    }

    /// Atomically replaces all routes with a new configuration set.
    pub fn reload(&self, routes: Vec<RouteConfig>) {
        let mut new_trie = RouteTrie::new();
        for config in routes {
            new_trie.insert(RouteRule::from(config));
        }

        let mut trie_guard = self.trie.write();
        *trie_guard = new_trie;
    }
}
