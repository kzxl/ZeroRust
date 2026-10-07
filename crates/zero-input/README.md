# `zero-input` ⚡🖱️⌨️

> Synthetic input injection, USB HID scancode mappings, anti-stuck key safety guard, and clipboard sync for ZConn in ZeroRust.

## Features
- **USB HID Scancodes**: Physical layout-independent scancode mappings.
- **SafetyGuard**: Prevents permanently stuck modifier keys (Ctrl, Alt, Shift, Win) upon window blur or connection loss.
- **Loop-Free Clipboard**: FNV-1a hash matching suppresses bi-directional echo loops.
- **Deterministic Testing**: `MockInjector` records all events for automated headless tests.
