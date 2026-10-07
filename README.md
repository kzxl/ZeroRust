# ZeroRust (`zero-rs`) ⚡🦀

> **Sovereign Industrial Automation, Hard Real-Time Fieldbus & Robotics Motion Control Framework for Rust.**  
> Proudly engineered as the hard real-time foundation of the [ZeroUniverse](https://github.com/kzxl/ZeroPlatform) ecosystem.

[![CI](https://github.com/kzxl/ZeroRust/actions/workflows/ci.yml/badge.svg)](https://github.com/kzxl/ZeroRust/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](crates/zero-core)

---

## 🌟 Vision & Architecture

Where **ZeroPlatform (.NET)** governs Desktop HMI, Distributed SCADA, High-Level Computer Vision, and AI-driven Process Supervision, **ZeroRust** commands the **Sub-Millisecond Edge, Hard Real-Time Robotics, Embedded I/O, and Industrial Protocols**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               ZeroPlatform (.NET 8/9/10)               │
                       │   Desktop HMI • SCADA • Distributed Cloud • Vision     │
                       └───────────────────────────▲────────────────────────────┘
                                                   │ zero-ipc (Shared Memory Ring Buffer)
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ZeroRust (`zero-rs`)                                     │
├───────────────────────────────┬────────────────────────────────┬───────────────────────────────┤
│          zero-core            │           zero-bus             │         zero-ethercat         │
│ • #![no_std] primitives      │ • Real-time CAN 2.0 / CAN-FD   │ • EtherCAT Master (ESM/CoE)   │
│ • Lock-free SPSC RingBuffer   │ • CANopen CiA 301 (NMT/SDO/PDO)│ • Distributed Clocks (DC) Sync│
│ • Zero-alloc Binary Parsing   │ • CiA 402 Servo Drive Profile  │ • Cyclic LRW Process Data     │
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│         zero-motion           │         zero-dynamics          │           zero-plc            │
│ • 6-DOF / SCARA / Cartesian   │ • Recursive Newton-Euler (RNEA)│ • IEC 61131-3 Bytecode VM     │
│ • Forward & Inverse Kinematics│ • Geometric Jacobian Matrix    │ • Standard FBs (TON, CTU, PID)│
│ • 7-Phase Jerk-Limited S-Curve│ • Sensorless Collision Detect  │ • Zero-Downtime Hot Reload    │
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│           zero-hal            │            zero-dsp            │          zero-modbus          │
│ • Digital I/O & Debouncing    │ • 2nd-Order Butterworth IIR    │ • Modbus RTU (Serial/RS485)   │
│ • 4x Quadrature Decoder       │ • Moving Average (Const-size)  │ • Modbus TCP (MBAP Header)    │
│ • Step/Dir DDA Pulse Train    │ • 1D Optimal Kalman Filter     │ • Tableless & Fast CRC16      │
│ • Hardware Mock Abstractions  │ • Radix-2 FFT Vibration Peak   │ • Zero-alloc PDU Serialization│
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│           zero-dds            │          zero-tinyai           │      zero-ipc & vision        │
│ • Pure Rust Zero-alloc CDR    │ • Int8 Quantized Forward Pass  │ • zero-ipc: 64B aligned SHM   │
│ • ROS 2 Standard Messages     │ • 1D Convolution Kernel        │ • zero-vision: Otsu / Sobel / │
│ • Micro-XRCE-DDS Client       │ • Predictive Anomaly AutoEnc   │   Connected Component Blobs   │
└───────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

---

## 📦 Crates Overview (13 Crates)

| Crate | Capabilities | Target Environments |
| :--- | :--- | :--- |
| **[`zero-core`](crates/zero-core)** | `#![no_std]` lock-free queues, cacheline-aligned SPSC ring buffer, zero-alloc byte slices, fixed-point math | Bare-metal MCUs (STM32, ESP32), RT-PREEMPT Linux, Windows |
| **[`zero-bus`](crates/zero-bus)** | Industrial Fieldbus engine: CAN 2.0/FD, CANopen CiA 301 master/slave, CiA 402 drive state machine (PPM, PVM, CSP) | Embedded Controllers, Industrial Gateways, RTOS |
| **[`zero-ethercat`](crates/zero-ethercat)** | Industrial Real-Time EtherCAT Master: Framing (0x88A4), ESM state machine, CoE mailbox, Distributed Clocks (DC) | Real-time multi-axis servo drives, EtherCAT terminals |
| **[`zero-motion`](crates/zero-motion)** | Robotics kinematics (6-Axis articulated, SCARA, Cartesian), Inverse Kinematics solvers, 7-phase jerk-limited S-curve motion profile | Robotic arms, CNC machines, multi-axis gantry stages |
| **[`zero-dynamics`](crates/zero-dynamics)** | Robotics dynamics: $O(N)$ Recursive Newton-Euler (RNEA), Mass matrix $M(q)$, Geometric Jacobian, sensorless collision detection, impedance/admittance | Collaborative robots (Cobots), compliance assembly |
| **[`zero-plc`](crates/zero-plc)** | Deterministic Soft-PLC runtime: IEC 61131-3 bytecode VM, standard FBs (`TON`, `TOF`, `CTU`, `PID_Compact`), zero-downtime hot-reload | Software PLC replacement, Edge machine controllers |
| **[`zero-hal`](crates/zero-hal)** | Hardware abstraction layer: Digital I/O debouncing, 4x Quadrature Encoder decoding with Index Z latching, Step/Dir DDA pulse generator | Stepper/Servo drives, Optical Encoders, Endstop sensors |
| **[`zero-dsp`](crates/zero-dsp)** | Real-time signal processing: Butterworth Low-Pass filter, Moving Average, 1D Kalman sensor fusion, Radix-2 FFT vibration spectrum analysis | Load cells, Pressure sensors, Bearing vibration analysis |
| **[`zero-modbus`](crates/zero-modbus)** | Industrial Modbus RTU (RS485) & Modbus TCP engine, zero-allocation PDU parsing, CRC16 checksum computation | PLCs, Remote I/O racks, Inverters, Energy meters |
| **[`zero-dds`](crates/zero-dds)** | Micro-ROS & ROS 2 direct link: zero-alloc OMG CDR serializer, ROS 2 standard messages (`Twist`, `Imu`), Micro-XRCE-DDS client | AMR/AGV mobile robotics, ROS 2 Nav2/MoveIt fleets |
| **[`zero-tinyai`](crates/zero-tinyai)** | Edge Neuromorphic TinyAI: Int8 quantized forward pass, Conv1D, Anomaly Detection AutoEncoder on vibration FFT | Bearing/spindle predictive maintenance on MCU |
| **[`zero-ipc`](crates/zero-ipc)** | Ultra-low-latency Shared Memory IPC bridge connecting ZeroRust edge controllers to ZeroPlatform (.NET) host/SCADA | Cross-process IPC, Windows FileMapping, POSIX SHM |
| **[`zero-vision`](crates/zero-vision)** | Embedded computer vision: Otsu auto-thresholding, Sobel edge filter, Connected Component Labeling (CCL) / Blob Analysis | Industrial sorting, PCB alignment, Edge optical inspection |

---

## 🚀 Quick Start Examples

### 1. Lock-free Shared Memory IPC Bridge (`zero-ipc`)

```rust
use zero_ipc::{ShmRingBuffer, MotionCommandPayload};

let mut shm_memory = [0u8; 4096];
let mut ring = ShmRingBuffer::init_new(&mut shm_memory, 16, 64).expect("SHM initialized");

let cmd = MotionCommandPayload {
    timestamp_ns: 1_000_000,
    command_type: 1, // Position mode
    target_positions: [1000, 2000, 3000, 0, 0, 0],
    target_velocities: [100, 200, 300, 0, 0, 0],
    controlword: 0x000F,
    reserved: 0,
};

ring.push(&cmd).expect("Command pushed to shared memory");
```

### 2. Embedded Vision Blob Extraction (`zero-vision`)

```rust
use zero_vision::{ImageBuffer, Blob, BlobAnalyzer, otsu_threshold, binarize};

let img: ImageBuffer<4096> = ImageBuffer::new(64, 64);
let th = otsu_threshold(&img);

let mut binary = ImageBuffer::new(64, 64);
binarize(&img, &mut binary, th, false);

let mut scratch = [0u16; 4096];
let mut blobs = [Blob::default(); 16];
let count = BlobAnalyzer::analyze(&binary, &mut scratch, &mut blobs);
println!("Detected {} foreground blobs on production line", count);
```

### 3. Step/Dir DDA Pulse Generator (`zero-hal`)

```rust
use zero_hal::stepdir::StepDirGenerator;

// 100 kHz base clock, commanding 10,000 steps/sec
let mut gen = StepDirGenerator::new(100_000);
gen.set_velocity(10_000, true);

let (step_pin, dir_pin) = gen.tick();
```

---

## 🛡️ Design Invariants

1. **Sub-Microsecond Determinism**: No heap allocation on critical real-time execution paths.
2. **`#![no_std]` First**: Fundamental algorithms and data structures run on bare-metal microcontrollers without operating system overhead.
3. **Safety & Zero Panic**: Rigorous error modeling via explicit Rust `Result<T, E>`.
4. **Symbiosis with ZeroUniverse**: Operates directly in tandem with `ZeroPlatform`'s high-level orchestrator.

---

## 📜 License

Licensed under the [MIT License](LICENSE).
