//! Bi-directional clipboard synchronization with loop-free echo suppression.

/// Data format of clipboard contents.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardFormat {
    /// Standard UTF-8 plain text string.
    PlainTextUtf8 = 0x01,
    /// PNG image format.
    PngImage = 0x02,
    /// Newline-delimited list of file paths.
    FileUriList = 0x03,
}

impl ClipboardFormat {
    /// Decodes format tag from byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::PlainTextUtf8),
            0x02 => Some(Self::PngImage),
            0x03 => Some(Self::FileUriList),
            _ => None,
        }
    }
}

/// Binary clipboard synchronization event payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardEvent {
    /// Content format.
    pub format: ClipboardFormat,
    /// FNV-1a 64-bit content hash for loop detection.
    pub content_hash: u64,
    /// Sequence counter.
    pub sequence: u32,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
}

impl ClipboardEvent {
    /// Creates a new clipboard event from raw payload and calculates its hash.
    pub fn new(format: ClipboardFormat, sequence: u32, payload: Vec<u8>) -> Self {
        let content_hash = compute_fnv1a_hash(&payload);
        Self {
            format,
            content_hash,
            sequence,
            payload,
        }
    }

    /// Serializes clipboard event into wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(17 + self.payload.len());
        b.push(self.format as u8);
        b.extend_from_slice(&self.content_hash.to_le_bytes());
        b.extend_from_slice(&self.sequence.to_le_bytes());
        b.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.payload);
        b
    }

    /// Deserializes clipboard event from wire bytes.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < 17 {
            return None;
        }
        let format = ClipboardFormat::from_u8(slice[0])?;
        let content_hash = u64::from_le_bytes([
            slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8],
        ]);
        let sequence = u32::from_le_bytes([slice[9], slice[10], slice[11], slice[12]]);
        let payload_len = u32::from_le_bytes([slice[13], slice[14], slice[15], slice[16]]) as usize;

        if slice.len() < 17 + payload_len {
            return None;
        }
        let payload = slice[17..17 + payload_len].to_vec();

        Some(Self {
            format,
            content_hash,
            sequence,
            payload,
        })
    }
}

/// Computes a fast 64-bit FNV-1a hash over byte slice.
pub fn compute_fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x00000100000001B3;
    const FNV_OFFSET_BASIS: u64 = 0xCBF29CE484222325;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Synchronizer that tracks clipboard content hashes to suppress infinite echo loops.
#[derive(Debug, Default)]
pub struct ClipboardSyncManager {
    last_local_hash: u64,
    last_remote_hash: u64,
    sequence: u32,
}

impl ClipboardSyncManager {
    /// Creates a new clipboard sync manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates local clipboard update. Returns event to transmit if content has mutated.
    pub fn on_local_change(
        &mut self,
        format: ClipboardFormat,
        payload: Vec<u8>,
    ) -> Option<ClipboardEvent> {
        let hash = compute_fnv1a_hash(&payload);
        if hash == self.last_local_hash || hash == self.last_remote_hash {
            return None; // Identical or echo from remote
        }
        self.last_local_hash = hash;
        self.sequence = self.sequence.wrapping_add(1);

        Some(ClipboardEvent {
            format,
            content_hash: hash,
            sequence: self.sequence,
            payload,
        })
    }

    /// Evaluates incoming remote clipboard event. Returns true if it should be applied locally.
    pub fn on_remote_event(&mut self, event: &ClipboardEvent) -> bool {
        if event.content_hash == self.last_local_hash {
            return false; // Echo suppression: this originated locally
        }
        self.last_remote_hash = event.content_hash;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipboard_serialization() {
        let text = b"Hello ZConn Sovereign Desktop".to_vec();
        let ev = ClipboardEvent::new(ClipboardFormat::PlainTextUtf8, 1, text);
        let bytes = ev.to_bytes();
        let parsed = ClipboardEvent::from_bytes(&bytes).expect("Parse clipboard");
        assert_eq!(ev, parsed);
    }

    #[test]
    fn test_clipboard_echo_suppression() {
        let mut mgr = ClipboardSyncManager::new();

        // 1. Local copy "Foo" -> emits event
        let ev = mgr
            .on_local_change(ClipboardFormat::PlainTextUtf8, b"Foo".to_vec())
            .expect("Emitted");
        assert_eq!(ev.payload, b"Foo");

        // 2. Incoming echo of same event -> ignored!
        assert!(!mgr.on_remote_event(&ev));

        // 3. New remote copy "Bar" -> accepted
        let remote_ev = ClipboardEvent::new(ClipboardFormat::PlainTextUtf8, 2, b"Bar".to_vec());
        assert!(mgr.on_remote_event(&remote_ev));
    }
}
