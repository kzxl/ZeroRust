//! Safe Hardware Abstraction Layer for DIGIC 4 / 5D Mark II in ZeroRust.
//!
//! Provides RAII synchronization primitives, critical sections,
//! VRAM surfaces, and atomic support on ARMv5TE.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod sys;

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};

// ------------------------------------------------------------------------
// Critical Section & Atomics Implementation for ARMv5TE (ARM946E-S)
// ------------------------------------------------------------------------
#[cfg(feature = "critical-section")]
struct D4CriticalSection;

#[cfg(feature = "critical-section")]
critical_section::set_impl!(D4CriticalSection);

#[cfg(all(feature = "critical-section", target_arch = "arm"))]
unsafe impl critical_section::Impl for D4CriticalSection {
    // `nomem` must NOT be used here: these asm blocks double as compiler memory barriers, so loads and
    // stores inside the critical section cannot be reordered across the IRQ disable/enable. The
    // ARM946E-S is single-core, so no hardware barrier (and no `dmb`, absent on ARMv5) is required.
    unsafe fn acquire() -> critical_section::RawRestoreState {
        let old_cpsr: u32;
        core::arch::asm!(
            "mrs {0}, CPSR",
            "orr r1, {0}, #0xC0", // Disable IRQ & FIQ
            "msr CPSR_c, r1",
            out(reg) old_cpsr,
            out("r1") _,
            options(nostack)
        );
        old_cpsr
    }

    unsafe fn release(restore_state: critical_section::RawRestoreState) {
        core::arch::asm!(
            "mrs r1, CPSR",
            "bic r1, r1, #0xC0",
            "and {0}, {0}, #0xC0",
            "orr r1, r1, {0}",
            "msr CPSR_c, r1",
            in(reg) restore_state,
            out("r1") _,
            options(nostack)
        );
    }
}

#[cfg(all(feature = "critical-section", not(target_arch = "arm")))]
unsafe impl critical_section::Impl for D4CriticalSection {
    unsafe fn acquire() -> critical_section::RawRestoreState {
        0
    }
    unsafe fn release(_restore_state: critical_section::RawRestoreState) {}
}

// ------------------------------------------------------------------------
// Safe RAII Semaphore
// ------------------------------------------------------------------------
pub struct D4Semaphore {
    raw: *mut sys::Semaphore,
}

unsafe impl Send for D4Semaphore {}
unsafe impl Sync for D4Semaphore {}

impl D4Semaphore {
    /// Creates a new named DryOS semaphore.
    ///
    /// # Safety
    /// `name` must be a null-terminated C string in static memory.
    pub unsafe fn new(name: *const u8, initial_unlocked: bool) -> Self {
        let starts = if initial_unlocked { 1 } else { 0 };
        let sem = sys::create_named_semaphore(name, starts);
        Self { raw: sem }
    }

    pub fn acquire(&self, timeout_ms: i32) -> bool {
        if self.raw.is_null() {
            return false;
        }
        unsafe { sys::take_semaphore(self.raw, timeout_ms) == 0 }
    }

    pub fn release(&self) {
        if !self.raw.is_null() {
            unsafe {
                sys::give_semaphore(self.raw);
            }
        }
    }
}

// ------------------------------------------------------------------------
// Thread-Safe Mutex using DryOS Semaphore
// ------------------------------------------------------------------------
pub struct D4Mutex<T> {
    sem: D4Semaphore,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for D4Mutex<T> {}
unsafe impl<T: Send> Sync for D4Mutex<T> {}

pub struct D4MutexGuard<'a, T> {
    mutex: &'a D4Mutex<T>,
}

impl<T> D4Mutex<T> {
    pub const unsafe fn from_raw(sem_ptr: *mut sys::Semaphore, data: T) -> Self {
        Self {
            sem: D4Semaphore { raw: sem_ptr },
            data: UnsafeCell::new(data),
        }
    }

    pub fn lock(&self) -> Option<D4MutexGuard<'_, T>> {
        if self.sem.acquire(0) {
            Some(D4MutexGuard { mutex: self })
        } else {
            None
        }
    }
}

impl<'a, T> Deref for D4MutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for D4MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T> Drop for D4MutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.sem.release();
    }
}

// ------------------------------------------------------------------------
// VRAM Surface Wrapper
// ------------------------------------------------------------------------
pub struct VramSurface<'a> {
    buffer: &'a mut [u8],
    width: usize,
    height: usize,
}

impl<'a> VramSurface<'a> {
    /// Creates a surface wrapper over camera VRAM.
    ///
    /// # Safety
    /// `raw_ptr` must be valid for `width * height * 2` bytes and 4-byte aligned.
    pub unsafe fn from_raw(raw_ptr: *mut u8, width: usize, height: usize) -> Self {
        let size = width.saturating_mul(height).saturating_mul(2);
        Self {
            buffer: core::slice::from_raw_parts_mut(raw_ptr, size),
            width,
            height,
        }
    }

    #[inline(always)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline(always)]
    pub fn put_pixel_fast(&mut self, x: usize, y: usize, color: u8) {
        let idx = (y * self.width + x) * 2;
        if idx + 1 < self.buffer.len() {
            self.buffer[idx + 1] = color;
        }
    }
}
