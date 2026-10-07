//! Shared Memory binary layout definitions compatible with ZeroPlatform (.NET C#).

use core::sync::atomic::AtomicU64;

/// Magic identifier for ZeroUniverse Shared Memory: "ZERO" in ASCII (0x5A45524F).
pub const IPC_MAGIC: u32 = 0x5A45_524F;

/// Current IPC protocol version.
pub const IPC_VERSION: u32 = 1;

/// Cacheline-aligned (64-byte) Shared Memory Ring Buffer Header.
///
/// Designed with strict binary layout matching C# `MemoryMarshal` / `Unsafe`:
/// ```csharp
/// [StructLayout(LayoutKind.Sequential, Pack = 8)]
/// public struct ShmHeader {
///     public uint Magic;
///     public uint Version;
///     public uint Capacity;
///     public uint SlotSize;
///     public ulong WriteHead;
///     public ulong ReadTail;
///     public uint Flags;
///     // 28 bytes padding to 64 bytes
/// }
/// ```
#[repr(C)]
pub struct ShmHeader {
    /// Magic number (0x5A45524F).
    pub magic: u32,
    /// Protocol version.
    pub version: u32,
    /// Total number of fixed-size slots in the ring buffer.
    pub capacity: u32,
    /// Size of each slot in bytes.
    pub slot_size: u32,
    /// Atomic monotonic write sequence counter.
    pub write_head: AtomicU64,
    /// Atomic monotonic read sequence counter.
    pub read_tail: AtomicU64,
    /// Status flags (Bit 0: initialized, Bit 1: overflow detected).
    pub flags: u32,
    /// Padding to fill exactly 64 bytes (1 CPU cacheline).
    pub padding: [u8; 28],
}

impl ShmHeader {
    /// Initializes a header in-place with capacity and slot size.
    pub fn init(&mut self, capacity: u32, slot_size: u32) {
        self.magic = IPC_MAGIC;
        self.version = IPC_VERSION;
        self.capacity = capacity;
        self.slot_size = slot_size;
        self.write_head = AtomicU64::new(0);
        self.read_tail = AtomicU64::new(0);
        self.flags = 1; // initialized
        self.padding = [0u8; 28];
    }

    /// Verifies the validity of the header.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.magic == IPC_MAGIC
            && self.version == IPC_VERSION
            && self.capacity > 0
            && self.slot_size > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_header_size_and_alignment() {
        assert_eq!(core::mem::size_of::<ShmHeader>(), 64);
        assert_eq!(core::mem::align_of::<ShmHeader>(), 8);
    }
}
