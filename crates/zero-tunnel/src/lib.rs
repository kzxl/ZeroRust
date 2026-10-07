//! # ZeroTunnel
//!
//! Low-latency multiplexed UDP transport, RFC 8489 STUN client, and E2EE security for ZConn in ZeroRust:
//! - `ZProto` binary multiplexed datagram protocol over UDP (channels: Control, Input, Video, Audio, Bulk)
//! - Pure-Rust RFC 8489 STUN Binding Request/Response parser (`XOR-MAPPED-ADDRESS`) and NAT classification
//! - ChaCha20 authenticated symmetric stream cipher with sequence-derived nonces
//! - `FrameJitterBuffer` with the "Always-Latest" frame drop policy to prevent interactive input lag

#![warn(missing_docs)]

pub mod crypto;
pub mod jitter;
pub mod proto;
pub mod stun;

pub use crypto::{chacha20_block, chacha20_xor, CryptoSession, KEY_BYTES, NONCE_BYTES, TAG_BYTES};
pub use jitter::{FrameJitterBuffer, JitterAction};
pub use proto::{
    ZProtoChannel, ZProtoHeader, ZProtoPacket, ZProtoPacketType, ZPROTO_MAGIC, ZPROTO_VERSION,
};
pub use stun::{NatType, StunMessage, STUN_MAGIC_COOKIE};
