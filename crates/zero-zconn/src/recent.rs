//! Recent connections manager and address book store for ZConn.

use crate::config::QualityPreset;

/// Metadata record for a previously connected remote desktop session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecentSession {
    /// 9-digit machine identifier or direct "IP:Port".
    pub target_id: String,
    /// User-defined friendly alias (e.g., "Office PC", "Accounting Server").
    pub alias: String,
    /// Monotonic timestamp in seconds of the last connection.
    pub last_connected_at: u64,
    /// Quality preset preferred for this connection.
    pub preferred_preset: QualityPreset,
    /// Whether this session is pinned to the top of the address book.
    pub is_favorite: bool,
    /// FNV-1a checksum of the desktop thumbnail snapshot.
    pub thumbnail_hash: u64,
}

impl RecentSession {
    /// Creates a new recent session entry.
    pub fn new(target_id: impl Into<String>, alias: impl Into<String>, now_secs: u64) -> Self {
        Self {
            target_id: target_id.into(),
            alias: alias.into(),
            last_connected_at: now_secs,
            preferred_preset: QualityPreset::Balanced,
            is_favorite: false,
            thumbnail_hash: 0,
        }
    }
}

/// In-memory store and manager for recent sessions history.
#[derive(Debug, Default)]
pub struct RecentStore {
    sessions: Vec<RecentSession>,
    max_capacity: usize,
}

impl RecentStore {
    /// Creates a new store with a capacity ceiling (e.g. 50 entries).
    pub fn new(max_capacity: usize) -> Self {
        Self {
            sessions: Vec::new(),
            max_capacity: max_capacity.max(5),
        }
    }

    /// Records or updates a session connection.
    pub fn record_connection(
        &mut self,
        target_id: &str,
        alias: Option<&str>,
        preset: QualityPreset,
        now_secs: u64,
    ) {
        if let Some(pos) = self.sessions.iter().position(|s| s.target_id == target_id) {
            let mut existing = self.sessions.remove(pos);
            existing.last_connected_at = now_secs;
            existing.preferred_preset = preset;
            if let Some(a) = alias {
                existing.alias = a.to_string();
            }
            self.sessions.insert(0, existing);
        } else {
            let new_entry = RecentSession {
                target_id: target_id.to_string(),
                alias: alias.unwrap_or(target_id).to_string(),
                last_connected_at: now_secs,
                preferred_preset: preset,
                is_favorite: false,
                thumbnail_hash: 0,
            };
            self.sessions.insert(0, new_entry);
        }

        if self.sessions.len() > self.max_capacity {
            // Trim oldest non-favorite entry
            if let Some(trim_idx) = self.sessions.iter().rposition(|s| !s.is_favorite) {
                self.sessions.remove(trim_idx);
            }
        }
    }

    /// Toggles the favorite / pinned status of a target.
    pub fn toggle_favorite(&mut self, target_id: &str) -> bool {
        if let Some(s) = self.sessions.iter_mut().find(|s| s.target_id == target_id) {
            s.is_favorite = !s.is_favorite;
            s.is_favorite
        } else {
            false
        }
    }

    /// Removes an entry from the recent connections history.
    pub fn remove(&mut self, target_id: &str) -> bool {
        if let Some(pos) = self.sessions.iter().position(|s| s.target_id == target_id) {
            self.sessions.remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns all sessions sorted by favorites first, then by last connected timestamp descending.
    pub fn list_sorted(&self) -> Vec<RecentSession> {
        let mut sorted = self.sessions.clone();
        sorted.sort_by(|a, b| {
            b.is_favorite
                .cmp(&a.is_favorite)
                .then_with(|| b.last_connected_at.cmp(&a.last_connected_at))
        });
        sorted
    }

    /// Total number of stored sessions.
    pub fn count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recent_store_ordering_and_favorites() {
        let mut store = RecentStore::new(10);
        store.record_connection("912 345 678", Some("PC 1"), QualityPreset::Balanced, 100);
        store.record_connection(
            "111 222 333",
            Some("PC 2"),
            QualityPreset::UltraLowLatency,
            200,
        );
        store.record_connection(
            "555 666 777",
            Some("PC 3"),
            QualityPreset::HighFidelity,
            300,
        );

        let list = store.list_sorted();
        assert_eq!(list[0].target_id, "555 666 777"); // Most recent first
        assert_eq!(list[1].target_id, "111 222 333");
        assert_eq!(list[2].target_id, "912 345 678");

        // Pin PC 1 as favorite -> it should jump to position 0!
        assert!(store.toggle_favorite("912 345 678"));
        let updated = store.list_sorted();
        assert_eq!(updated[0].target_id, "912 345 678"); // Favorite pinned first!
        assert_eq!(updated[1].target_id, "555 666 777");
    }

    #[test]
    fn test_recent_store_capacity_trimming() {
        let mut store = RecentStore::new(5);
        for i in 0..10 {
            store.record_connection(
                &format!("ID-{}", i),
                None,
                QualityPreset::Balanced,
                i as u64,
            );
        }
        assert!(store.count() <= 5);
    }
}
