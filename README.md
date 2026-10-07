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
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│           zero-hal            │            zero-dsp            │          zero-modbus          │
│ • Digital I/O & Debouncing    │ • 2nd-Order Butterworth IIR    │ • Modbus RTU (Serial/RS485)   │
│ • 4x Quadrature Decoder       │ • Moving Average (Const-size)  │ • Modbus TCP (MBAP Header)    │
│ • Step/Dir DDA Pulse Train    │ • 1D Optimal Kalman Filter     │ • Tableless & Fast CRC16      │
│ • Hardware Mock Abstractions  │ • Radix-2 FFT Vibration Peak   │ • Zero-alloc PDU Serialization│
└───────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

---

## 📦 Crates Overview

| Crate | Capabilities | Target Environments |
| :--- | :--- | :--- |
| **[`zero-core`](crates/zero-core)** | `#![no_std]` lock-free queues, cacheline-aligned SPSC ring buffer, zero-alloc byte slices, fixed-point math | Bare-metal MCUs (STM32, ESP32), RT-PREEMPT Linux, Windows |
| **[`zero-bus`](crates/zero-bus)** | Industrial Fieldbus engine: CAN 2.0/FD, CANopen CiA 301 master/slave, CiA 402 drive state machine (PPM, PVM, CSP) | Embedded Controllers, Industrial Gateways, RTOS |
| **[`zero-motion`](crates/zero-motion)** | Robotics kinematics (6-Axis articulated, SCARA, Cartesian), Inverse Kinematics solvers, 7-phase jerk-limited S-curve motion profile | Robotic arms, CNC machines, multi-axis gantry stages |
| **[`zero-hal`](crates/zero-hal)** | Hardware abstraction layer: Digital I/O debouncing, 4x Quadrature Encoder decoding with Index Z latching, Step/Dir DDA pulse generator | Stepper/Servo drives, Optical Encoders, Endstop sensors |
| **[`zero-dsp`](crates/zero-dsp)** | Real-time signal processing: Butterworth Low-Pass filter, Moving Average, 1D Kalman sensor fusion, Radix-2 FFT vibration spectrum analysis | Load cells, Pressure sensors, Bearing vibration analysis |
| **[`zero-modbus`](crates/zero-modbus)** | Industrial Modbus RTU (RS485) & Modbus TCP engine, zero-allocation PDU parsing, CRC16 checksum computation | PLCs, Remote I/O racks, Inverters, Energy meters |

---

## 🚀 Quick Start Examples

### 1. Lock-free SPSC Ring Buffer (`zero-core`)

```rust
use zero_core::ring_buffer::SpscRingBuffer;

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

drive.process_controlword(ControlWord::SHUTDOWN);
drive.process_controlword(ControlWord::SWITCH_ON);
drive.process_controlword(ControlWord::ENABLE_OPERATION);

assert_eq!(drive.current_state(), DriveState::OperationEnabled);
```

### 3. Step/Dir DDA Pulse Generator (`zero-hal`)

```rust
use zero_hal::stepdir::StepDirGenerator;

// 100 kHz base clock, commanding 10,000 steps/sec
let mut gen = StepDirGenerator::new(100_000);
gen.set_velocity(10_000, true);

// Execute in real-time timer interrupt (10us cycle)
let (step_pin, dir_pin) = gen.tick();
```

### 4. Vibration Spectrum Analysis with Radix-2 FFT (`zero-dsp`)

```rust
use zero_dsp::fft::{Complex64, Radix2Fft};

let mut signal = [Complex64::default(); 256];
// ... fill signal with vibration accelerometer data ...
Radix2Fft::transform(&mut signal).expect("FFT transformed");

let mut magnitudes = [0.0; 128];
Radix2Fft::compute_magnitudes(&signal, &mut magnitudes).unwrap();
let dominant_frequency_bin = Radix2Fft::find_peak_bin(&magnitudes);
```

### 5. Modbus RTU Frame Parsing (`zero-modbus`)

```rust
use zero_modbus::rtu::ModbusRtu;

let frame = [0x01, 0x03, 0x00, 0x02, 0x00, 0x04, 0xE5, 0xC9];
if let Ok((slave, req)) = ModbusRtu::parse_frame(&frame) {
    println!("Slave: {}, Request: {:?}", slave, req);
}
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
