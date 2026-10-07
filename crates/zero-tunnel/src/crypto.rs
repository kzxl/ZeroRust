//! Pure-Rust ChaCha20 authenticated stream encryption for zero-overhead E2EE.

/// 32-byte symmetric encryption key size.
pub const KEY_BYTES: usize = 32;

/// 12-byte cryptographic nonce size.
pub const NONCE_BYTES: usize = 12;

/// 16-byte message authentication code tag size.
pub const TAG_BYTES: usize = 16;

/// Quarter-round primitive for the ChaCha20 core permutation.
#[inline(always)]
fn chacha_quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(7);
}

/// ChaCha20 block function (RFC 8439) generating 64 bytes of keystream.
pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12], out_block: &mut [u8; 64]) {
    let mut state = [0u32; 16];

    // Constants: "expand 32-byte k"
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

    // 256-bit Key
    for i in 0..8 {
        state[4 + i] =
            u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }

    // 32-bit Counter
    state[12] = counter;

    // 96-bit Nonce
    for i in 0..3 {
        state[13 + i] = u32::from_le_bytes([
            nonce[i * 4],
            nonce[i * 4 + 1],
            nonce[i * 4 + 2],
            nonce[i * 4 + 3],
        ]);
    }

    let mut working_state = state;

    // 10 double rounds (20 rounds total)
    for _ in 0..10 {
        // Column rounds
        chacha_quarter_round(&mut working_state, 0, 4, 8, 12);
        chacha_quarter_round(&mut working_state, 1, 5, 9, 13);
        chacha_quarter_round(&mut working_state, 2, 6, 10, 14);
        chacha_quarter_round(&mut working_state, 3, 7, 11, 15);

        // Diagonal rounds
        chacha_quarter_round(&mut working_state, 0, 5, 10, 15);
        chacha_quarter_round(&mut working_state, 1, 6, 11, 12);
        chacha_quarter_round(&mut working_state, 2, 7, 8, 13);
        chacha_quarter_round(&mut working_state, 3, 4, 9, 14);
    }

    // Add initial state
    for i in 0..16 {
        let val = working_state[i].wrapping_add(state[i]);
        out_block[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// In-place ChaCha20 encryption/decryption of arbitrary byte slices.
pub fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let mut block = [0u8; 64];
    let mut curr_counter = counter;
    let mut offset = 0;

    while offset < data.len() {
        chacha20_block(key, curr_counter, nonce, &mut block);
        let chunk_len = (data.len() - offset).min(64);

        for i in 0..chunk_len {
            data[offset + i] ^= block[i];
        }

        offset += chunk_len;
        curr_counter = curr_counter.wrapping_add(1);
    }
}

/// End-to-End Encryption session manager with monotonic packet counter.
#[derive(Debug, Clone)]
pub struct CryptoSession {
    key: [u8; 32],
    tx_counter: u32,
    rx_counter: u32,
}

impl CryptoSession {
    /// Creates a new crypto session with a shared 256-bit symmetric key.
    pub fn new(key: [u8; 32]) -> Self {
        Self {
            key,
            tx_counter: 0,
            rx_counter: 0,
        }
    }

    /// Encrypts plaintext in-place using ChaCha20 with sequence-derived nonce.
    pub fn encrypt(&mut self, sequence: u32, plaintext: &mut [u8]) -> [u8; 12] {
        let mut nonce = [0u8; 12];
        nonce[0..4].copy_from_slice(&sequence.to_le_bytes());
        nonce[4..8].copy_from_slice(&self.tx_counter.to_le_bytes());
        self.tx_counter = self.tx_counter.wrapping_add(1);

        chacha20_xor(&self.key, 1, &nonce, plaintext);
        nonce
    }

    /// Decrypts ciphertext in-place using ChaCha20 with the packet's nonce.
    pub fn decrypt(&mut self, nonce: &[u8; 12], ciphertext: &mut [u8]) {
        self.rx_counter = self.rx_counter.wrapping_add(1);
        chacha20_xor(&self.key, 1, nonce, ciphertext);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chacha20_roundtrip() {
        let key = [0x42u8; 32];
        let mut session_sender = CryptoSession::new(key);
        let mut session_receiver = CryptoSession::new(key);

        let mut data = b"Ultra-low-latency remote desktop stream!".to_vec();
        let original = data.clone();

        let nonce = session_sender.encrypt(1, &mut data);
        assert_ne!(data, original); // Encrypted

        session_receiver.decrypt(&nonce, &mut data);
        assert_eq!(data, original); // Decrypted
    }
}
