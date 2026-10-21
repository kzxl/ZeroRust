//! # ZeroGateway Automated Verification Test Suite
//!
//! Validates routing Trie, atomic load balancing, token bucket limiter, JWT, and telemetry.

use std::sync::Arc;
use zero_gateway::auth::jwt::{hmac_sha256, sha256};
use zero_gateway::cluster::balancer::{LoadBalancer, LoadBalancingPolicy};
use zero_gateway::cluster::destination::Destination;
use zero_gateway::config::{DestinationConfig, RouteConfig};
use zero_gateway::pipeline::ratelimit::TokenBucket;
use zero_gateway::proxy::buffer_pool::BufferPool;
use zero_gateway::routing::rule::RouteRule;
use zero_gateway::routing::trie::RouteTrie;
use zero_gateway::telemetry::histogram::LatencyHistogram;

#[test]
fn test_prefix_trie_routing() {
    let mut trie = RouteTrie::new();

    trie.insert(RouteRule::from(RouteConfig {
        route_id: "orders-api".to_string(),
        match_path: "/api/v1/orders".to_string(),
        cluster_id: "cluster-orders".to_string(),
        strip_prefix: false,
        rate_limit_per_second: 1000,
        auth_required: false,
        is_active: true,
    }));

    trie.insert(RouteRule::from(RouteConfig {
        route_id: "root-api".to_string(),
        match_path: "/api".to_string(),
        cluster_id: "cluster-root".to_string(),
        strip_prefix: false,
        rate_limit_per_second: 500,
        auth_required: false,
        is_active: true,
    }));

    // Longest prefix match
    let match_order = trie.match_path("/api/v1/orders/123").expect("Must match orders");
    assert_eq!(match_order.route_id, "orders-api");

    let match_root = trie.match_path("/api/v2/users").expect("Must fallback to root");
    assert_eq!(match_root.route_id, "root-api");

    let match_none = trie.match_path("/other/resource");
    assert!(match_none.is_none());
}

#[test]
fn test_path_rewriting() {
    let rule = RouteRule::from(RouteConfig {
        route_id: "strip-api".to_string(),
        match_path: "/api/gateway".to_string(),
        cluster_id: "cluster-backend".to_string(),
        strip_prefix: true,
        rate_limit_per_second: 0,
        auth_required: false,
        is_active: true,
    });

    assert_eq!(rule.rewrite_path("/api/gateway/items/list"), "/items/list");
    assert_eq!(rule.rewrite_path("/api/gateway"), "/");
}

#[test]
fn test_round_robin_balancer() {
    let d1 = Arc::new(Destination::from(DestinationConfig {
        address: "10.0.0.1:8080".to_string(),
        weight: 1,
        role: "Active".to_string(),
        is_active: true,
    }));
    let d2 = Arc::new(Destination::from(DestinationConfig {
        address: "10.0.0.2:8080".to_string(),
        weight: 1,
        role: "Active".to_string(),
        is_active: true,
    }));

    let dests = vec![d1.clone(), d2.clone()];
    let balancer = LoadBalancer::new(LoadBalancingPolicy::RoundRobin);

    let sel1 = balancer.select(&dests).unwrap();
    let sel2 = balancer.select(&dests).unwrap();
    let sel3 = balancer.select(&dests).unwrap();

    assert_eq!(sel1.address, "10.0.0.1:8080");
    assert_eq!(sel2.address, "10.0.0.2:8080");
    assert_eq!(sel3.address, "10.0.0.1:8080");
}

#[test]
fn test_least_connections_balancer() {
    let d1 = Arc::new(Destination::from(DestinationConfig {
        address: "10.0.0.1:8080".to_string(),
        weight: 1,
        role: "Active".to_string(),
        is_active: true,
    }));
    let d2 = Arc::new(Destination::from(DestinationConfig {
        address: "10.0.0.2:8080".to_string(),
        weight: 1,
        role: "Active".to_string(),
        is_active: true,
    }));

    d1.record_start();
    d1.record_start(); // d1 has 2 in-flight
    d2.record_start(); // d2 has 1 in-flight

    let dests = vec![d1.clone(), d2.clone()];
    let balancer = LoadBalancer::new(LoadBalancingPolicy::LeastConnections);

    let selected = balancer.select(&dests).unwrap();
    assert_eq!(selected.address, "10.0.0.2:8080");
}

#[test]
fn test_token_bucket_rate_limiter() {
    let bucket = TokenBucket::new(2); // 2 tokens per sec
    assert!(bucket.allow());
    assert!(bucket.allow());
    assert!(!bucket.allow()); // 3rd must be rejected
}

#[test]
fn test_pure_sha256_and_hmac() {
    let key = b"my-secret-key";
    let data = b"hello-world";
    let sig1 = hmac_sha256(key, data);
    let sig2 = hmac_sha256(key, data);
    assert_eq!(sig1, sig2);

    let hash = sha256(b"abc");
    assert_eq!(hash.len(), 32);
}

#[test]
fn test_buffer_pool_reuse() {
    let pool = BufferPool::new(4, 8);
    {
        let mut buf = pool.acquire();
        buf[0] = 42;
        assert_eq!(buf[0], 42);
    } // buf dropped back to pool

    let buf2 = pool.acquire();
    assert_eq!(buf2.len(), 32 * 1024);
}

#[test]
fn test_latency_histogram_percentiles() {
    let hist = LatencyHistogram::new();
    for _ in 0..50 {
        hist.record(400); // 0.4 ms
    }
    for _ in 0..40 {
        hist.record(800); // 0.8 ms
    }
    for _ in 0..10 {
        hist.record(2000); // 2.0 ms
    }

    let (p50, p90, p99, avg) = hist.percentiles();
    assert!(p50 <= 0.5);
    assert!(p90 <= 1.0);
    assert!(p99 <= 2.5);
    assert!(avg > 0.0);
}
