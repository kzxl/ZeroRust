//! Lock-free Shared Memory Ring Buffer reader and writer.

use crate::layout::ShmHeader;
use core::sync::atomic::Ordering;
use zero_core::error::{ZeroError, ZeroResult};

/// Lock-free Shared Memory Ring Buffer controller over an arbitrary contiguous memory region.
pub struct ShmRingBuffer<'a> {
    raw_memory: &'a mut [u8],
}

impl<'a> ShmRingBuffer<'a> {
    /// Creates and initializes a new shared memory ring buffer in the given memory slice.
    pub fn init_new(memory: &'a mut [u8], capacity: u32, slot_size: u32) -> ZeroResult<Self> {
        let header_size = core::mem::size_of::<ShmHeader>();
        let total_required = header_size + (capacity as usize) * (slot_size as usize);

        if memory.len() < total_required {
            return Err(ZeroError::BufferOverflow);
        }

        // Initialize header
        let header_ptr = memory.as_mut_ptr() as *mut ShmHeader;
        unsafe {
            (*header_ptr).init(capacity, slot_size);
        }

        Ok(Self { raw_memory: memory })
    }

    /// Attaches to an already initialized shared memory ring buffer.
    pub fn attach(memory: &'a mut [u8]) -> ZeroResult<Self> {
        let header_size = core::mem::size_of::<ShmHeader>();
        if memory.len() < header_size {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let header_ptr = memory.as_ptr() as *const ShmHeader;
        let is_valid = unsafe { (*header_ptr).is_valid() };

        if !is_valid {
            return Err(ZeroError::InvalidArgument);
        }

        Ok(Self { raw_memory: memory })
    }

    #[inline]
    fn header(&self) -> &ShmHeader {
        unsafe { &*(self.raw_memory.as_ptr() as *const ShmHeader) }
    }

    /// Pushes a structured payload into the next available slot.
    pub fn push<T: Copy>(&mut self, item: &T) -> ZeroResult<()> {
        let item_size = core::mem::size_of::<T>();
        let (capacity, slot_size, head, tail) = {
            let header = self.header();
            if item_size > header.slot_size as usize {
                return Err(ZeroError::InvalidArgument);
            }
            let head = header.write_head.load(Ordering::Relaxed);
            let tail = header.read_tail.load(Ordering::Acquire);
            (
                header.capacity as u64,
                header.slot_size as usize,
                head,
                tail,
            )
        };

        if head.saturating_sub(tail) >= capacity {
            return Err(ZeroError::BufferFull);
        }

        let slot_index = (head % capacity) as usize;
        let header_size = core::mem::size_of::<ShmHeader>();
        let slot_offset = header_size + slot_index * slot_size;

        unsafe {
            let dest_ptr = self.raw_memory.as_mut_ptr().add(slot_offset) as *mut T;
            dest_ptr.write_unaligned(*item);
        }

        let header = self.header();
        header.write_head.store(head + 1, Ordering::Release);
        Ok(())
    }

    /// Pops the next structured payload from the ring buffer.
    pub fn pop<T: Copy>(&mut self) -> ZeroResult<Option<T>> {
        let item_size = core::mem::size_of::<T>();
        let (capacity, slot_size, head, tail) = {
            let header = self.header();
            if item_size > header.slot_size as usize {
                return Err(ZeroError::InvalidArgument);
            }
            let tail = header.read_tail.load(Ordering::Relaxed);
            let head = header.write_head.load(Ordering::Acquire);
            (
                header.capacity as u64,
                header.slot_size as usize,
                head,
                tail,
            )
        };

        if tail >= head {
            return Ok(None);
        }

        let slot_index = (tail % capacity) as usize;
        let header_size = core::mem::size_of::<ShmHeader>();
        let slot_offset = header_size + slot_index * slot_size;

        let item = unsafe {
            let src_ptr = self.raw_memory.as_ptr().add(slot_offset) as *const T;
            src_ptr.read_unaligned()
        };

        let header = self.header();
        header.read_tail.store(tail + 1, Ordering::Release);
        Ok(Some(item))
    }

    /// Returns the number of unread items in the ring buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let header = self.header();
        let head = header.write_head.load(Ordering::Acquire);
        let tail = header.read_tail.load(Ordering::Acquire);
        head.saturating_sub(tail) as usize
    }

    /// Returns true if there are no pending unread items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payloads::MotionCommandPayload;

    #[test]
    fn test_shm_ring_push_pop() {
        #[repr(align(8))]
        struct AlignedMem([u8; 1024]);
        let mut mem = AlignedMem([0u8; 1024]);
        let mut ring = ShmRingBuffer::init_new(&mut mem.0, 8, 64).unwrap();

        assert!(ring.is_empty());

        let cmd = MotionCommandPayload {
            timestamp_ns: 1_000_000,
            command_type: 1,
            target_positions: [100, 200, 300, 0, 0, 0],
            target_velocities: [10, 20, 30, 0, 0, 0],
            controlword: 0x000F,
            reserved: 0,
        };

        assert!(ring.push(&cmd).is_ok());
        assert_eq!(ring.len(), 1);

        let popped: Option<MotionCommandPayload> = ring.pop().unwrap();
        assert_eq!(popped, Some(cmd));
        assert!(ring.is_empty());
    }
}
