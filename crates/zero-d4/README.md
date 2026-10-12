# zero-d4 📷

> Safe Hardware Abstraction Layer (HAL) and raw DryOS RTOS bindings for **Canon DIGIC 4 / 5D Mark II** in **ZeroRust**.

Part of the **ZeroUniverse** ecosystem.

## Features

- **DryOS RTOS Bindings**: Low-level stubs for Task management, Semaphores, RecursiveLocks, and Memory allocation.
- **Strict ABI Verification**: Compile-time assertion of `struct Task` layout (0x50 bytes) exactly matching `src/tasks.h`.
- **Interrupt & Critical Section**: Single-core ARMv5TE compiler memory barriers for safe atomic operations without `nomem` clobber hazards.
- **Safe VRAM Surfaces**: Bounded pixel manipulation wrappers over camera LiveView memory buffers.

## License

Licensed under MIT. Authored by **Phong Vo & The ZeroUniverse Team**.
