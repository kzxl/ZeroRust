//! # Cacheline-Aligned Pre-Allocated Buffer Pool
//!
//! Complies with AgentOption Memory Standards (memory-performance.md).
//! 64-byte CPU cacheline alignment, zero-allocation buffer acquisition with RAII Drop.

use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub const BUFFER_SIZE: usize = 32 * 1024; // 32 KB per chunk

/// 64-byte aligned memory chunk eliminating false sharing and cache-line crossing.
#[repr(C, align(64))]
pub struct AlignedBuffer {
    pub data: [u8; BUFFER_SIZE],
}

impl AlignedBuffer {
    pub fn new() -> Self {
        Self {
            data: [0u8; BUFFER_SIZE],
        }
    }
}

impl Default for AlignedBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe buffer pool minimizing heap allocations on the hot proxy streaming path.
pub struct BufferPool {
    pool: Mutex<Vec<Box<AlignedBuffer>>>,
    max_capacity: usize,
}

impl BufferPool {
    pub fn new(initial_capacity: usize, max_capacity: usize) -> Arc<Self> {
        let mut buffers = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            buffers.push(Box::new(AlignedBuffer::new()));
        }

        Arc::new(Self {
            pool: Mutex::new(buffers),
            max_capacity,
        })
    }

    /// Acquires an aligned buffer from the pool, allocating only if the pool is exhausted.
    pub fn acquire(self: &Arc<Self>) -> BufferGuard {
        let buffer = {
            let mut guard = self.pool.lock();
            guard.pop().unwrap_or_else(|| Box::new(AlignedBuffer::new()))
        };

        BufferGuard {
            buffer: Some(buffer),
            pool: Arc::clone(self),
        }
    }

    fn release(&self, buffer: Box<AlignedBuffer>) {
        let mut guard = self.pool.lock();
        if guard.len() < self.max_capacity {
            guard.push(buffer);
        }
    }
}

/// RAII Guard that returns the borrowed buffer back to the pool on drop.
pub struct BufferGuard {
    buffer: Option<Box<AlignedBuffer>>,
    pool: Arc<BufferPool>,
}

impl Deref for BufferGuard {
    type Target = [u8; BUFFER_SIZE];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Safe: buffer is always Some during active guard lifetime
        match &self.buffer {
            Some(b) => &b.data,
            None => unreachable!(),
        }
    }
}

impl DerefMut for BufferGuard {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.buffer {
            Some(b) => &mut b.data,
            None => unreachable!(),
        }
    }
}

impl Drop for BufferGuard {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.pool.release(buf);
        }
    }
}
