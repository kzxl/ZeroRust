//! Quadrature Encoder Decoder with 4x resolution and Index pulse detection.

/// Rotation direction of the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Clockwise / Forward rotation.
    Forward,
    /// Counter-clockwise / Reverse rotation.
    Reverse,
    /// Stationary / Stopped.
    Stationary,
}

/// 4x Quadrature Encoder state decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuadratureDecoder {
    last_state: u8,
    position: i64,
    index_latched: bool,
    latched_index_pos: i64,
    last_delta: i8,
}

impl QuadratureDecoder {
    /// Lookup table for 4x quadrature decoding:
    /// Index = (last_state << 2) | current_state
    /// Values: 0 = no change, +1 = forward, -1 = reverse, 2 = error/glitch
    const TRANSITION_TABLE: [i8; 16] = [
        0, -1, 1, 0, // 00 -> 00, 01, 10, 11
        1, 0, 0, -1, // 01 -> 00, 01, 10, 11
        -1, 0, 0, 1, // 10 -> 00, 01, 10, 11
        0, 1, -1, 0, // 11 -> 00, 01, 10, 11
    ];

    /// Creates a new quadrature decoder initialized to zero position.
    pub const fn new() -> Self {
        Self {
            last_state: 0,
            position: 0,
            index_latched: false,
            latched_index_pos: 0,
            last_delta: 0,
        }
    }

    /// Feeds new states of Channel A, Channel B, and optional Index pulse.
    pub fn update(&mut self, ch_a: bool, ch_b: bool, index: bool) -> i64 {
        let current_state = ((ch_a as u8) << 1) | (ch_b as u8);
        let table_index = ((self.last_state & 0x03) << 2) | (current_state & 0x03);
        let delta = Self::TRANSITION_TABLE[table_index as usize];

        self.last_state = current_state;
        self.last_delta = delta;
        self.position += delta as i64;

        if index && !self.index_latched {
            self.index_latched = true;
            self.latched_index_pos = self.position;
        } else if !index {
            self.index_latched = false;
        }

        self.position
    }

    /// Returns the accumulated encoder position in counts.
    #[inline]
    pub const fn position(&self) -> i64 {
        self.position
    }

    /// Resets the encoder position and internal state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.last_state = 0;
        self.position = 0;
        self.index_latched = false;
        self.latched_index_pos = 0;
        self.last_delta = 0;
    }

    /// Resets the encoder position to zero or a specific count without changing state.
    #[inline]
    pub fn reset_position(&mut self, pos: i64) {
        self.position = pos;
    }

    /// Returns the detected direction of rotation.
    #[inline]
    pub fn direction(&self) -> Direction {
        if self.last_delta > 0 {
            Direction::Forward
        } else if self.last_delta < 0 {
            Direction::Reverse
        } else {
            Direction::Stationary
        }
    }

    /// Returns the latched position at the most recent Index Z pulse.
    #[inline]
    pub const fn latched_index_position(&self) -> i64 {
        self.latched_index_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrature_forward_sequence() {
        let mut enc = QuadratureDecoder::new();
        assert_eq!(enc.position(), 0);

        // Forward sequence starting from 00: 00 -> 10 -> 11 -> 01 -> 00
        enc.update(true, false, false); // 10: delta = +1
        assert_eq!(enc.position(), 1);
        assert_eq!(enc.direction(), Direction::Forward);

        enc.update(true, true, false); // 11: delta = +1
        assert_eq!(enc.position(), 2);

        enc.update(false, true, false); // 01: delta = +1
        assert_eq!(enc.position(), 3);

        enc.update(false, false, false); // 00: delta = +1
        assert_eq!(enc.position(), 4);
    }

    #[test]
    fn test_quadrature_reverse_sequence() {
        let mut enc = QuadratureDecoder::new();
        // Reverse sequence starting from 00: 00 -> 01 -> 11 -> 10 -> 00
        enc.update(false, true, false); // 01: delta = -1
        assert_eq!(enc.position(), -1);
        assert_eq!(enc.direction(), Direction::Reverse);

        enc.update(true, true, false); // 11: delta = -1
        assert_eq!(enc.position(), -2);

        enc.update(true, false, false); // 10: delta = -1
        assert_eq!(enc.position(), -3);

        enc.update(false, false, false); // 00: delta = -1
        assert_eq!(enc.position(), -4);
    }
}
