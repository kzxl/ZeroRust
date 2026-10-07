# zero-ethercat

Industrial Real-Time EtherCAT Master engine for the `ZeroRust` embedded ecosystem.

## Features
- **Zero Heap Allocation**: `#![no_std]` capable with stack/static-only buffers and zero allocations on cyclic paths.
- **EtherCAT Framing**: Standard EtherType `0x88A4`, datagram encoding/decoding (`APRD`, `APWR`, `FPRD`, `FPWR`, `BRD`, `BWR`, `LRD`, `LWR`, `LRW`), Working Counter (WKC) verification.
- **EtherCAT State Machine (ESM)**: Complete transition validation and register mapping for `Init` -> `PreOp` -> `SafeOp` -> `Op`, AL Control/Status (0x0120/0x0130), and AL Status Code diagnostics.
- **CANopen over EtherCAT (CoE)**: Mailbox Header, CoE Service headers, expedited SDO Download and Upload transactions bridging CiA 402 drive profiles.
- **Distributed Clocks (DC)**: Nanosecond-precision propagation delay compensation and cyclic SYNC0/SYNC1 interrupt pulse alignment.
