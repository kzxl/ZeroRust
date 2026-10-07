//! Jitter buffer and "Always-Latest" frame drop governor for real-time video streaming.

use std::collections::HashMap;

/// Result of pushing a video tile fragment into the jitter buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitterAction {
    /// Fragment accepted, frame is still assembling.
    Buffering {
        /// Target frame sequence number.
        frame_id: u32,
        /// Number of fragments received so far for this tile.
        received: usize,
        /// Total fragments required.
        total: usize,
    },
    /// All fragments for this tile/frame arrived and are ready for presentation.
    TileComplete {
        /// Target frame sequence number.
        frame_id: u32,
        /// Tile grid index.
        tile_index: u16,
        /// Reassembled binary payload.
        payload: Vec<u8>,
    },
    /// Fragment discarded because it belongs to an outdated/stale frame.
    DroppedStale {
        /// Outdated frame sequence number.
        frame_id: u32,
        /// Highest sequence rendered by the presentation engine.
        latest_frame: u32,
    },
}

/// Jitter buffer enforcing an "Always-Latest" frame policy to guarantee minimal input lag.
#[derive(Debug, Default)]
pub struct FrameJitterBuffer {
    /// Highest frame ID rendered so far.
    latest_rendered_frame_id: u32,
    /// Pending tile fragments indexed by `(frame_id, tile_index)`.
    pending_fragments: HashMap<(u32, u16), Vec<Option<Vec<u8>>>>,
}

impl FrameJitterBuffer {
    /// Creates a new frame jitter buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests an incoming tile fragment datagram.
    pub fn push_fragment(
        &mut self,
        frame_id: u32,
        tile_index: u16,
        chunk_index: u8,
        total_chunks: u8,
        data: Vec<u8>,
    ) -> JitterAction {
        // Drop rule: if frame_id is older than latest rendered frame, drop immediately!
        if frame_id < self.latest_rendered_frame_id {
            return JitterAction::DroppedStale {
                frame_id,
                latest_frame: self.latest_rendered_frame_id,
            };
        }

        // Advance to newer frame: clean up older pending frames
        if frame_id > self.latest_rendered_frame_id {
            self.pending_fragments
                .retain(|&(fid, _), _| fid >= frame_id);
        }

        if total_chunks <= 1 {
            // Single fragment: immediately complete!
            return JitterAction::TileComplete {
                frame_id,
                tile_index,
                payload: data,
            };
        }

        let entry = self
            .pending_fragments
            .entry((frame_id, tile_index))
            .or_insert_with(|| vec![None; total_chunks as usize]);

        let idx = chunk_index as usize;
        if idx < entry.len() {
            entry[idx] = Some(data);
        }

        // Check if all chunks have arrived
        let received_count = entry.iter().filter(|c| c.is_some()).count();
        if received_count == entry.len() {
            let mut complete_payload = Vec::new();
            for c in entry.drain(..).flatten() {
                complete_payload.extend_from_slice(&c);
            }
            self.pending_fragments.remove(&(frame_id, tile_index));
            JitterAction::TileComplete {
                frame_id,
                tile_index,
                payload: complete_payload,
            }
        } else {
            JitterAction::Buffering {
                frame_id,
                received: received_count,
                total: entry.len(),
            }
        }
    }

    /// Marks a frame as rendered, discarding any pending tiles from earlier frames.
    pub fn mark_frame_rendered(&mut self, frame_id: u32) {
        if frame_id > self.latest_rendered_frame_id {
            self.latest_rendered_frame_id = frame_id;
            self.pending_fragments
                .retain(|&(fid, _), _| fid >= frame_id);
        }
    }

    /// Returns the latest rendered frame ID.
    pub fn latest_frame_id(&self) -> u32 {
        self.latest_rendered_frame_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitter_buffer_fragment_reassembly() {
        let mut jb = FrameJitterBuffer::new();

        // Push chunk 0 of 2
        let a1 = jb.push_fragment(10, 0, 0, 2, b"Part1-".to_vec());
        assert!(matches!(a1, JitterAction::Buffering { .. }));

        // Push chunk 1 of 2
        let a2 = jb.push_fragment(10, 0, 1, 2, b"Part2".to_vec());
        match a2 {
            JitterAction::TileComplete { payload, .. } => {
                assert_eq!(payload, b"Part1-Part2");
            }
            _ => panic!("Expected TileComplete"),
        }

        // Mark frame 10 rendered
        jb.mark_frame_rendered(10);

        // Frame 9 arriving late should be dropped
        let a3 = jb.push_fragment(9, 0, 0, 1, b"Late".to_vec());
        assert!(matches!(a3, JitterAction::DroppedStale { .. }));
    }
}
