# zero-plc

Deterministic Soft-PLC runtime conforming to IEC 61131-3 for `ZeroRust`.

## Features
- **Zero Dynamic Allocation**: `#![no_std]` capable with static bounded process images and execution stacks.
- **IEC 61131-3 Memory Model**: Fast bit, byte, word, and dword/float addressing across `%I` (Inputs), `%Q` (Outputs), and `%M` (Markers).
- **Bytecode Virtual Machine**: Deterministic Instruction List (IL) / Structured Text (ST) stack interpreter executing in $100\,\mu\text{s} - 1\,\text{ms}$ scan slices.
- **Standard Function Blocks**:
  - Timers: `TON` (On-Delay), `TOF` (Off-Delay), `TP` (Pulse).
  - Counters: `CTU` (Count Up), `CTD` (Count Down).
  - Closed-Loop Control: `PID_Compact` with anti-windup clamping and low-pass derivative filtering.
- **Zero-Downtime Hot-Reload**: Live program swapping between scan cycles without zeroing `%M` state memory or disturbing machine drives.
