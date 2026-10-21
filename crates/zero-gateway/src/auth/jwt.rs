//! # Pure Rust Zero-Dependency JWT & HMAC-SHA256 Validator
//!
//! Complies with AgentOption Zero-Dependency & Safety Standards.
//! RFC 6234 SHA-256 and RFC 2104 HMAC implemented in pure Rust without OpenSSL.

use crate::error::GatewayError;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Standard JWT payload claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    #[serde(default)]
    pub sub: String,
    #[serde(default)]
    pub exp: u64,
    #[serde(default)]
    pub role: String,
}

/// Validates a Bearer JWT token against the given secret key.
pub fn validate_jwt(token: &str, secret: &[u8]) -> Result<JwtClaims, GatewayError> {
    let clean_token = token.trim_start_matches("Bearer ").trim();
    let parts: Vec<&str> = clean_token.split('.').collect();
    if parts.len() != 3 {
        return Err(GatewayError::AuthTokenInvalid {
            reason: "Token must contain 3 dot-separated segments".to_string(),
        });
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Compute expected signature: HMAC_SHA256(secret, "header.payload")
    let signed_data = format!("{header_b64}.{payload_b64}");
    let expected_sig = hmac_sha256(secret, signed_data.as_bytes());

    let provided_sig = decode_base64_url(signature_b64).map_err(|e| GatewayError::AuthTokenInvalid {
        reason: format!("Signature base64 decode failed: {e}"),
    })?;

    // Constant-time signature comparison to prevent timing attacks
    if !constant_time_eq(&expected_sig, &provided_sig) {
        return Err(GatewayError::AuthTokenInvalid {
            reason: "Signature verification failed".to_string(),
        });
    }

    // Decode and parse claims payload
    let payload_bytes = decode_base64_url(payload_b64).map_err(|e| GatewayError::AuthTokenInvalid {
        reason: format!("Payload base64 decode failed: {e}"),
    })?;

    let claims: JwtClaims = serde_json::from_slice(&payload_bytes).map_err(|e| {
        GatewayError::AuthTokenInvalid {
            reason: format!("Invalid claims JSON: {e}"),
        }
    })?;

    // Check expiration timestamp if provided
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if claims.exp > 0 && claims.exp < now {
        return Err(GatewayError::AuthTokenInvalid {
            reason: format!("Token expired at {} (current: {})", claims.exp, now),
        });
    }

    Ok(claims)
}

#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Pure Rust HMAC-SHA256 calculation (RFC 2104).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let block_size = 64;
    let mut key_block = [0u8; 64];

    if key.len() > block_size {
        let hashed_key = sha256(key);
        key_block[..32].copy_from_slice(&hashed_key);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut o_key_pad = [0x5cu8; 64];
    let mut i_key_pad = [0x36u8; 64];

    for i in 0..block_size {
        o_key_pad[i] ^= key_block[i];
        i_key_pad[i] ^= key_block[i];
    }

    let mut inner_input = Vec::with_capacity(block_size + data.len());
    inner_input.extend_from_slice(&i_key_pad);
    inner_input.extend_from_slice(data);
    let inner_hash = sha256(&inner_input);

    let mut outer_input = [0u8; 64 + 32];
    outer_input[..64].copy_from_slice(&o_key_pad);
    outer_input[64..].copy_from_slice(&inner_hash);
    sha256(&outer_input)
}

/// Pure Rust SHA-256 hash calculation (FIPS 180-4).
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0]; let mut b = h[1]; let mut c = h[2]; let mut d = h[3];
        let mut e = h[4]; let mut f = h[5]; let mut g = h[6]; let mut h_val = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_val.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_val = g; g = f; f = e; e = d.wrapping_add(temp1);
            d = c; c = b; b = a; a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(h_val);
    }

    let mut result = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// Simple Base64-URL decoder without padding.
fn decode_base64_url(input: &str) -> Result<Vec<u8>, String> {
    let mut base64_str = input.replace('-', "+").replace('_', "/");
    while !base64_str.len().is_multiple_of(4) {
        base64_str.push('=');
    }

    let table = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let bytes = base64_str.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b0 = bytes[i]; let b1 = bytes[i + 1];
        let b2 = bytes[i + 2]; let b3 = bytes[i + 3];

        let v0 = table.iter().position(|&c| c == b0).ok_or("Invalid char")? as u32;
        let v1 = table.iter().position(|&c| c == b1).ok_or("Invalid char")? as u32;
        let v2 = if b2 == b'=' { 0 } else { table.iter().position(|&c| c == b2).ok_or("Invalid char")? as u32 };
        let v3 = if b3 == b'=' { 0 } else { table.iter().position(|&c| c == b3).ok_or("Invalid char")? as u32 };

        let triple = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;

        out.push(((triple >> 16) & 0xFF) as u8);
        if b2 != b'=' { out.push(((triple >> 8) & 0xFF) as u8); }
        if b3 != b'=' { out.push((triple & 0xFF) as u8); }

        i += 4;
    }
    Ok(out)
}
