//! Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer.
//!
//! Provides deterministic FIFO buffering without dynamic memory allocation or OS locks.
//! Safe for concurrent use between hard real-time interrupt handlers, RT-PREEMPT threads,
//! and worker loops.

use crate::error::{ZeroError, ZeroResult};
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Cache-line padding to prevent false sharing between reader and writer cores.
#[repr(align(64))]
struct CachePadded<T>(T);

/// A lock-free, zero-allocation Bounded Single-Producer Single-Consumer (SPSC) Ring Buffer.
pub struct SpscRingBuffer<T, const CAPACITY: usize> {
    buffer: [UnsafeCell<MaybeUninit<T>>; CAPACITY],
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send, const CAPACITY: usize> Send for SpscRingBuffer<T, CAPACITY> {}
unsafe impl<T: Send, const CAPACITY: usize> Sync for SpscRingBuffer<T, CAPACITY> {}

impl<T, const CAPACITY: usize> SpscRingBuffer<T, CAPACITY> {
    /// Creates a new empty `SpscRingBuffer`.
    pub const fn new() -> Self {
        assert!(CAPACITY > 1, "RingBuffer capacity must be greater than 1");

        // Safety: uninitialized array of UnsafeCell<MaybeUninit<T>> is safe to construct
        let buffer = unsafe {
            MaybeUninit::<[UnsafeCell<MaybeUninit<T>>; CAPACITY]>::uninit().assume_init()
        };

        Self {
            buffer,
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
        }
    }

    /// Returns the maximum capacity of the ring buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    /// Returns the current number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        if head >= tail {
            head - tail
        } else {
            CAPACITY - (tail - head)
        }
    }

    /// Returns true if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.0.load(Ordering::Acquire) == self.tail.0.load(Ordering::Acquire)
    }

    /// Returns true if the buffer is completely full.
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        (head + 1) % CAPACITY == tail
    }

    /// Enqueues an item into the buffer.
    ///
    /// # Errors
    /// Returns `ZeroError::BufferFull` if the buffer is full.
    pub fn push(&self, item: T) -> ZeroResult<()> {
        let head = self.head.0.load(Ordering::Relaxed);
        let next_head = (head + 1) % CAPACITY;
        let tail = self.tail.0.load(Ordering::Acquire);

        if next_head == tail {
            return Err(ZeroError::BufferFull);
        }

        unsafe {
            let slot = self.buffer[head].get();
            (*slot).write(item);
        }

        self.head.0.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Dequeues an item from the buffer.
    ///
    /// Returns `None` if the buffer is empty.
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        let item = unsafe {
            let slot = self.buffer[tail].get();
            (*slot).assume_init_read()
        };

        let next_tail = (tail + 1) % CAPACITY;
        self.tail.0.store(next_tail, Ordering::Release);
        Some(item)
    }
}

impl<T, const CAPACITY: usize> Default for SpscRingBuffer<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAPACITY: usize> Drop for SpscRingBuffer<T, CAPACITY> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_basic() {
        let ring: SpscRingBuffer<u32, 8> = SpscRingBuffer::new();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);

        assert!(ring.push(10).is_ok());
        assert!(ring.push(20).is_ok());
        assert_eq!(ring.len(), 2);
        assert!(!ring.is_empty());

        assert_eq!(ring.pop(), Some(10));
        assert_eq!(ring.pop(), Some(20));
        assert_eq!(ring.pop(), None);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_buffer_full() {
        let ring: SpscRingBuffer<usize, 4> = SpscRingBuffer::new();
        // With capacity 4, can store at most 3 items to differentiate full from empty in simple ring
        assert!(ring.push(1).is_ok());
        assert!(ring.push(2).is_ok());
        assert!(ring.push(3).is_ok());
        assert_eq!(ring.push(4), Err(ZeroError::BufferFull));

        assert_eq!(ring.pop(), Some(1));
        assert!(ring.push(5).is_ok());
    }
}
