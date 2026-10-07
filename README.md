# ZeroRust (`zero-rs`) ⚡🦀

> **Sovereign Industrial Automation, Hard Real-Time Fieldbus & Robotics Motion Control Framework for Rust.**  
> Proudly engineered as the hard real-time foundation of the [ZeroUniverse](https://github.com/kzxl/ZeroPlatform) ecosystem.

[![CI](https://github.com/kzxl/ZeroRust/actions/workflows/ci.yml/badge.svg)](https://github.com/kzxl/ZeroRust/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](crates/zero-core)

---

## 🌟 Vision & Architecture

Where **ZeroPlatform (.NET)** governs Desktop HMI, Distributed SCADA, High-Level Computer Vision, and AI-driven Process Supervision, **ZeroRust** commands the **Sub-Millisecond Edge, Hard Real-Time Robotics, and Embedded Fieldbus**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               ZeroPlatform (.NET 8/9/10)               │
                       │   Desktop HMI • SCADA • Distributed Cloud • Vision     │
                       └───────────────────────────▲────────────────────────────┘
                                                   │ IPC / ZeroMQ / gRPC
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ZeroRust (`zero-rs`)                                     │
├───────────────────────────────┬────────────────────────────────┬───────────────────────────────┤
│          zero-core            │           zero-bus             │          zero-motion          │
│ • #![no_std] primitives      │ • Real-time CAN 2.0 / CAN-FD   │ • 6-DOF / SCARA / Cartesian   │
│ • Lock-free SPSC RingBuffer   │ • CANopen CiA 301 (NMT/SDO/PDO)│ • Forward & Inverse Kinematics│
│ • Zero-alloc Binary Parsing   │ • CiA 402 Servo Drive Profile  │ • Geometric Jacobian Matrix   │
│ • Q16.16/Q32.32 Fixed Point   │ • Micro-second Cyclic Sync     │ • 7-Phase Jerk-Limited S-Curve│
└───────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

---

## 📦 Crates Overview

| Crate | Capabilities | Target Environments |
| :--- | :--- | :--- |
| **[`zero-core`](crates/zero-core)** | `#![no_std]` lock-free queues, cacheline-aligned SPSC ring buffer, zero-alloc byte slices, fixed-point math | Bare-metal MCUs (STM32, ESP32), RT-PREEMPT Linux, Windows |
| **[`zero-bus`](crates/zero-bus)** | Industrial Fieldbus engine: CAN 2.0/FD, CANopen CiA 301 master/slave, CiA 402 drive state machine (PPM, PVM, CSP) | Embedded Controllers, Industrial Gateways, RTOS |
| **[`zero-motion`](crates/zero-motion)** | Robotics kinematics (6-Axis articulated, SCARA, Cartesian), Inverse Kinematics solvers, 7-phase jerk-limited S-curve motion profile | Robotic arms, CNC machines, multi-axis gantry stages |

---

## 🚀 Quick Start

### 1. Lock-free SPSC Ring Buffer (`zero-core`)

```rust
use zero_core::ring_buffer::SpscRingBuffer;

// Deterministic, lock-free queue with fixed capacity 64, zero heap allocation
static QUEUE: SpscRingBuffer<u32, 64> = SpscRingBuffer::new();

fn main() {
    assert!(QUEUE.push(42).is_ok());
    assert_eq!(QUEUE.pop(), Some(42));
}
```

### 2. CANopen CiA 402 Servo Drive State Machine (`zero-bus`)

```rust
use zero_bus::cia402::{Cia402Drive, ControlWord, DriveState, OperationMode};

let mut drive = Cia402Drive::new(1); // Node ID 1
drive.set_target_mode(OperationMode::CyclicSynchronousPosition);

// Transition from SwitchOnDisabled -> ReadyToSwitchOn -> SwitchedOn -> OperationEnabled
drive.process_controlword(ControlWord::SHUTDOWN);
drive.process_controlword(ControlWord::SWITCH_ON);
drive.process_controlword(ControlWord::ENABLE_OPERATION);

assert_eq!(drive.current_state(), DriveState::OperationEnabled);
```

### 3. Jerk-Limited 7-Phase S-Curve Profile (`zero-motion`)

```rust
use zero_motion::scurve::{SCurvePlanner, SCurveConstraints};

let constraints = SCurveConstraints {
    max_velocity: 1000.0,     // units/s
    max_acceleration: 2000.0, // units/s^2
    max_jerk: 5000.0,         // units/s^3
};

let planner = SCurvePlanner::new(constraints);
let trajectory = planner.plan(0.0, 500.0).expect("Trajectory planned");

// Sample at deterministic cyclic intervals (e.g., 1ms fieldbus cycle)
let sample = trajectory.sample(0.250);
println!("Pos: {}, Vel: {}, Acc: {}", sample.position, sample.velocity, sample.acceleration);
```

---

## 🛡️ Design Invariants

1. **Sub-Microsecond Determinism**: No heap allocation on critical real-time execution paths.
2. **`#![no_std]` First**: Fundamental algorithms and data structures run on bare-metal microcontrollers without operating system overhead.
3. **Safety & Zero Panic**: Rigorous error modeling via explicit Rust `Result<T, E>`.
4. **Symbiosis with ZeroUniverse**: Sits seamlessly below `ZeroPlatform`'s high-level orchestrator.

---

## 📜 License

Licensed under the [MIT License](LICENSE).
