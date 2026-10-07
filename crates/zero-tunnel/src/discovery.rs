//! Serverless local network auto-discovery protocol via UDP broadcast beacons.

use std::collections::HashMap;
use std::net::SocketAddr;

/// Local discovery beacon payload broadcasted by active ZConn hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryBeacon {
    /// 9-digit device identifier.
    pub device_id: u32,
    /// UDP listening port.
    pub port: u16,
    /// Host computer name.
    pub hostname: String,
    /// Operating system string.
    pub os_type: String,
    /// Preferred quality preset byte.
    pub preset: u8,
}

impl DiscoveryBeacon {
    /// Serializes beacon into binary wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(&self.device_id.to_le_bytes());
        b.extend_from_slice(&self.port.to_le_bytes());
        b.push(self.preset);

        let host_bytes = self.hostname.as_bytes();
        let host_len = (host_bytes.len().min(32)) as u8;
        b.push(host_len);
        b.extend_from_slice(&host_bytes[..host_len as usize]);

        let os_bytes = self.os_type.as_bytes();
        let os_len = (os_bytes.len().min(16)) as u8;
        b.push(os_len);
        b.extend_from_slice(&os_bytes[..os_len as usize]);

        b
    }

    /// Deserializes beacon from binary wire bytes.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < 7 {
            return None;
        }
        let device_id = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
        let port = u16::from_le_bytes([slice[4], slice[5]]);
        let preset = slice[6];

        let mut offset = 7;
        if slice.len() <= offset {
            return None;
        }
        let host_len = slice[offset] as usize;
        offset += 1;
        if slice.len() < offset + host_len {
            return None;
        }
        let hostname = String::from_utf8_lossy(&slice[offset..offset + host_len]).to_string();
        offset += host_len;

        if slice.len() <= offset {
            return None;
        }
        let os_len = slice[offset] as usize;
        offset += 1;
        if slice.len() < offset + os_len {
            return None;
        }
        let os_type = String::from_utf8_lossy(&slice[offset..offset + os_len]).to_string();

        Some(Self {
            device_id,
            port,
            hostname,
            os_type,
            preset,
        })
    }
}

/// Discovered peer on the local subnet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredPeer {
    /// Remote socket endpoint.
    pub addr: SocketAddr,
    /// 9-digit device ID.
    pub device_id: u32,
    /// Host name.
    pub hostname: String,
    /// Operating system.
    pub os_type: String,
    /// Quality preset.
    pub preset: u8,
    /// Monotonic timestamp in ms of last received beacon.
    pub last_seen_ms: u64,
}

/// In-memory local peer discovery cache.
#[derive(Debug, Default)]
pub struct DiscoveryManager {
    peers: HashMap<u32, DiscoveredPeer>,
}

impl DiscoveryManager {
    /// Creates a new discovery manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes an incoming beacon packet received over UDP broadcast.
    pub fn on_beacon_received(
        &mut self,
        sender_addr: SocketAddr,
        beacon_bytes: &[u8],
        now_ms: u64,
    ) -> Option<DiscoveredPeer> {
        let beacon = DiscoveryBeacon::from_bytes(beacon_bytes)?;
        let mut actual_addr = sender_addr;
        actual_addr.set_port(beacon.port);

        let peer = DiscoveredPeer {
            addr: actual_addr,
            device_id: beacon.device_id,
            hostname: beacon.hostname,
            os_type: beacon.os_type,
            preset: beacon.preset,
            last_seen_ms: now_ms,
        };

        self.peers.insert(beacon.device_id, peer.clone());
        Some(peer)
    }

    /// Prunes stale peers that haven't sent a beacon in `timeout_ms`.
    pub fn prune_stale_peers(&mut self, now_ms: u64, timeout_ms: u64) {
        self.peers
            .retain(|_, p| now_ms.saturating_sub(p.last_seen_ms) <= timeout_ms);
    }

    /// Returns list of currently active LAN peers.
    pub fn active_peers(&self) -> Vec<DiscoveredPeer> {
        self.peers.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_beacon_serialization() {
        let b = DiscoveryBeacon {
            device_id: 912_345_678,
            port: 21118,
            hostname: "Accounting-PC".to_string(),
            os_type: "windows".to_string(),
            preset: 1,
        };
        let bytes = b.to_bytes();
        let parsed = DiscoveryBeacon::from_bytes(&bytes).expect("Parse beacon");
        assert_eq!(b, parsed);
    }

    #[test]
    fn test_discovery_manager_lifecycle() {
        let mut mgr = DiscoveryManager::new();
        let beacon = DiscoveryBeacon {
            device_id: 111_222_333,
            port: 21118,
            hostname: "Office-Server".to_string(),
            os_type: "linux".to_string(),
            preset: 2,
        };

        let addr: SocketAddr = "192.168.1.50:54321".parse().unwrap();
        let peer = mgr
            .on_beacon_received(addr, &beacon.to_bytes(), 1000)
            .expect("Peer discovered");
        assert_eq!(peer.device_id, 111_222_333);
        assert_eq!(peer.addr.port(), 21118);
        assert_eq!(mgr.active_peers().len(), 1);

        // Pruning within timeout keeps peer
        mgr.prune_stale_peers(3000, 5000);
        assert_eq!(mgr.active_peers().len(), 1);

        // Pruning past timeout removes peer
        mgr.prune_stale_peers(7000, 5000);
        assert_eq!(mgr.active_peers().len(), 0);
    }
}
