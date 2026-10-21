//! # High-Performance Prefix Radix Trie Router
//!
//! Sub-microsecond path matching with O(K) complexity bounded by path segments.
//! Strictly zero runtime panic, zero unwrap.

use super::rule::RouteRule;
use std::sync::Arc;

/// A node within the routing Prefix Trie.
#[derive(Debug, Clone)]
pub struct TrieNode {
    pub segment: String,
    pub children: Vec<TrieNode>,
    pub rule: Option<Arc<RouteRule>>,
    pub is_wildcard: bool,
}

impl TrieNode {
    pub fn new(segment: String) -> Self {
        let is_wildcard = segment == "*" || segment == "**";
        Self {
            segment,
            children: Vec::new(),
            rule: None,
            is_wildcard,
        }
    }
}

/// The root Trie structure holding compiled routes.
#[derive(Debug, Clone)]
pub struct RouteTrie {
    root: TrieNode,
}

impl Default for RouteTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(String::new()),
        }
    }

    /// Inserts a new route rule into the Trie.
    pub fn insert(&mut self, rule: RouteRule) {
        if !rule.is_active {
            return;
        }

        let rule_arc = Arc::new(rule);
        let path = rule_arc.match_path.trim_matches('/');
        let segments: Vec<&str> = if path.is_empty() {
            Vec::new()
        } else {
            path.split('/').collect()
        };

        let mut current = &mut self.root;

        for seg in segments {
            let child_idx = current.children.iter().position(|c| c.segment == seg);
            match child_idx {
                Some(idx) => {
                    current = match current.children.get_mut(idx) {
                        Some(child) => child,
                        None => return,
                    };
                }
                None => {
                    let new_node = TrieNode::new(seg.to_string());
                    current.children.push(new_node);
                    let last_idx = current.children.len().saturating_sub(1);
                    current = match current.children.get_mut(last_idx) {
                        Some(child) => child,
                        None => return,
                    };
                }
            }
        }

        current.rule = Some(rule_arc);
    }

    /// Finds the best matching route rule for the given request path.
    /// Returns the matched route rule using longest prefix matching.
    pub fn match_path(&self, req_path: &str) -> Option<Arc<RouteRule>> {
        let clean_path = req_path.split('?').next().unwrap_or(req_path);
        let trimmed = clean_path.trim_matches('/');

        let mut best_match: Option<Arc<RouteRule>> = self.root.rule.clone();

        if trimmed.is_empty() {
            return best_match;
        }

        let segments: Vec<&str> = trimmed.split('/').collect();
        let mut current = &self.root;

        for seg in segments {
            // Find matching child segment or wildcard
            let mut matched_child = None;
            for child in &current.children {
                if child.segment == seg || child.is_wildcard {
                    matched_child = Some(child);
                    break;
                }
            }

            match matched_child {
                Some(child) => {
                    current = child;
                    if let Some(ref r) = current.rule {
                        best_match = Some(Arc::clone(r));
                    }
                }
                None => break,
            }
        }

        best_match
    }
}
