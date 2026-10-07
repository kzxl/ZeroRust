//! MTU datagram packetization and fragment reassembly.

/// Maximum payload capacity per UDP datagram (leaving room for headers and crypto tags).
pub const MAX_UDP_PAYLOAD_BYTES: usize = 1160;

/// Fragment header prepended to every UDP tile datagram chunk.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileFragmentHeader {
    /// Frame sequence identifier.
    pub frame_id: u32,
    /// Linear tile index within the frame.
    pub tile_index: u16,
    /// Index of this fragment chunk (0-based).
    pub chunk_index: u8,
    /// Total fragments comprising this complete tile.
    pub total_chunks: u8,
    /// Length of the fragment payload following this header.
    pub chunk_len: u16,
}

impl TileFragmentHeader {
    /// Size of binary fragment header in bytes (10 bytes).
    pub const SIZE: usize = 10;

    /// Serializes fragment header into a byte array.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut b = [0u8; Self::SIZE];
        b[0..4].copy_from_slice(&self.frame_id.to_le_bytes());
        b[4..6].copy_from_slice(&self.tile_index.to_le_bytes());
        b[6] = self.chunk_index;
        b[7] = self.total_chunks;
        b[8..10].copy_from_slice(&self.chunk_len.to_le_bytes());
        b
    }

    /// Deserializes fragment header from a byte slice.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < Self::SIZE {
            return None;
        }
        let frame_id = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
        let tile_index = u16::from_le_bytes([slice[4], slice[5]]);
        let chunk_index = slice[6];
        let total_chunks = slice[7];
        let chunk_len = u16::from_le_bytes([slice[8], slice[9]]);

        Some(Self {
            frame_id,
            tile_index,
            chunk_index,
            total_chunks,
            chunk_len,
        })
    }
}

/// Packetizes tile data into MTU-safe chunks.
pub struct TilePacketizer;

impl TilePacketizer {
    /// Splits an encoded tile into fragments appended directly into an existing vector.
    pub fn packetize_into(
        frame_id: u32,
        tile_index: u16,
        tile_data: &[u8],
        out: &mut Vec<Vec<u8>>,
    ) {
        if tile_data.is_empty() {
            return;
        }

        let total_chunks = tile_data.len().div_ceil(MAX_UDP_PAYLOAD_BYTES) as u8;

        for chunk_idx in 0..total_chunks {
            let start = (chunk_idx as usize) * MAX_UDP_PAYLOAD_BYTES;
            let end = (start + MAX_UDP_PAYLOAD_BYTES).min(tile_data.len());
            let chunk_slice = &tile_data[start..end];

            let header = TileFragmentHeader {
                frame_id,
                tile_index,
                chunk_index: chunk_idx,
                total_chunks,
                chunk_len: chunk_slice.len() as u16,
            };

            let mut packet = Vec::with_capacity(TileFragmentHeader::SIZE + chunk_slice.len());
            packet.extend_from_slice(&header.to_bytes());
            packet.extend_from_slice(chunk_slice);
            out.push(packet);
        }
    }

    /// Splits an encoded tile into a vector of UDP datagram fragments.
    pub fn packetize(frame_id: u32, tile_index: u16, tile_data: &[u8]) -> Vec<Vec<u8>> {
        let total_chunks = tile_data.len().div_ceil(MAX_UDP_PAYLOAD_BYTES);
        let mut packets = Vec::with_capacity(total_chunks);
        Self::packetize_into(frame_id, tile_index, tile_data, &mut packets);
        packets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_packetization() {
        let sample_data = vec![0xAB; 2500]; // Should split into 3 chunks (1160 + 1160 + 180)
        let packets = TilePacketizer::packetize(42, 5, &sample_data);

        assert_eq!(packets.len(), 3);

        let h0 = TileFragmentHeader::from_bytes(&packets[0]).expect("Parse header 0");
        assert_eq!(h0.frame_id, 42);
        assert_eq!(h0.tile_index, 5);
        assert_eq!(h0.chunk_index, 0);
        assert_eq!(h0.total_chunks, 3);
        assert_eq!(h0.chunk_len as usize, 1160);

        let h2 = TileFragmentHeader::from_bytes(&packets[2]).expect("Parse header 2");
        assert_eq!(h2.chunk_index, 2);
        assert_eq!(h2.chunk_len as usize, 180);
    }
}
