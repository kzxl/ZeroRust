//! Pure-Rust RFC 8489 / RFC 5389 STUN client and NAT type classifier.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// STUN Magic Cookie as specified by RFC 5389 (`0x2112A442`).
pub const STUN_MAGIC_COOKIE: u32 = 0x2112A442;

/// STUN Binding Request message type (`0x0001`).
pub const STUN_BINDING_REQUEST: u16 = 0x0001;

/// STUN Binding Success Response message type (`0x0101`).
pub const STUN_BINDING_RESPONSE: u16 = 0x0101;

/// STUN Attribute: XOR-MAPPED-ADDRESS (`0x0020`).
pub const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

/// NAT Behavior Classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// Direct public IP / Open Internet (no NAT).
    OpenInternet,
    /// Full-cone NAT (100% P2P hole punch success).
    FullCone,
    /// Restricted-cone NAT (IP-restricted).
    RestrictedCone,
    /// Port-restricted cone NAT.
    PortRestricted,
    /// Symmetric NAT (requires relay fallback or port prediction).
    Symmetric,
}

/// Pure Rust STUN message parser and builder.
pub struct StunMessage;

impl StunMessage {
    /// Generates a 20-byte STUN Binding Request packet with a given 12-byte transaction ID.
    pub fn build_binding_request(transaction_id: [u8; 12]) -> [u8; 20] {
        let mut msg = [0u8; 20];
        msg[0..2].copy_from_slice(&STUN_BINDING_REQUEST.to_be_bytes());
        msg[2..4].copy_from_slice(&0u16.to_be_bytes()); // Message length = 0
        msg[4..8].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        msg[8..20].copy_from_slice(&transaction_id);
        msg
    }

    /// Parses a STUN Binding Success Response and extracts the public reflexive `SocketAddr`.
    pub fn parse_binding_response(packet: &[u8]) -> Option<SocketAddr> {
        if packet.len() < 20 {
            return None;
        }

        let msg_type = u16::from_be_bytes([packet[0], packet[1]]);
        if msg_type != STUN_BINDING_RESPONSE {
            return None;
        }

        let msg_len = u16::from_be_bytes([packet[2], packet[3]]) as usize;
        let magic_cookie = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);
        if magic_cookie != STUN_MAGIC_COOKIE {
            return None;
        }

        let mut offset = 20;
        let end = (20 + msg_len).min(packet.len());

        while offset + 4 <= end {
            let attr_type = u16::from_be_bytes([packet[offset], packet[offset + 1]]);
            let attr_len = u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]) as usize;
            offset += 4;

            if offset + attr_len > packet.len() {
                break;
            }

            if attr_type == ATTR_XOR_MAPPED_ADDRESS && attr_len >= 8 {
                let family = packet[offset + 1];
                if family == 0x01 {
                    // IPv4
                    let xor_port = u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]);
                    let port = xor_port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);

                    let xor_ip = u32::from_be_bytes([
                        packet[offset + 4],
                        packet[offset + 5],
                        packet[offset + 6],
                        packet[offset + 7],
                    ]);
                    let ip = xor_ip ^ STUN_MAGIC_COOKIE;

                    return Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(ip)), port));
                }
            }

            // Align attribute to 4-byte boundary
            offset += (attr_len + 3) & !3;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stun_binding_request_and_response() {
        let tx_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let req = StunMessage::build_binding_request(tx_id);
        assert_eq!(req.len(), 20);
        assert_eq!(u16::from_be_bytes([req[0], req[1]]), STUN_BINDING_REQUEST);

        // Construct synthetic STUN response with XOR-MAPPED-ADDRESS for 203.0.113.45:54321
        let mut resp = Vec::new();
        resp.extend_from_slice(&STUN_BINDING_RESPONSE.to_be_bytes());
        resp.extend_from_slice(&12u16.to_be_bytes()); // attr_len = 12
        resp.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        resp.extend_from_slice(&tx_id);

        // Attribute XOR-MAPPED-ADDRESS
        resp.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
        resp.extend_from_slice(&8u16.to_be_bytes()); // length
        resp.push(0); // reserved
        resp.push(0x01); // IPv4

        let port: u16 = 54321;
        let xor_port = port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);
        resp.extend_from_slice(&xor_port.to_be_bytes());

        let ip_u32 = u32::from(Ipv4Addr::new(203, 0, 113, 45));
        let xor_ip = ip_u32 ^ STUN_MAGIC_COOKIE;
        resp.extend_from_slice(&xor_ip.to_be_bytes());

        let addr = StunMessage::parse_binding_response(&resp).expect("Parse STUN response");
        assert_eq!(addr, "203.0.113.45:54321".parse::<SocketAddr>().unwrap());
    }
}
