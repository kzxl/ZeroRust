# `zero-tunnel` ⚡🌐

> Low-latency multiplexed UDP transport, RFC 8489 STUN client, NAT hole punching, and E2EE security for ZConn in ZeroRust.

## Features
- **ZProto Framing**: 20-byte compact header multiplexing 5 logical channels (Control, Input, Video, Audio, Bulk).
- **Pure-Rust RFC 8489 STUN**: Resolves public reflexive IP and port with zero external C dependencies.
- **E2EE ChaCha20**: Military-grade authenticated symmetric stream encryption with monotonic sequence nonces.
- **Always-Latest Jitter Buffer**: Drops stale video frames immediately to eliminate input lag.
