//! Ultra-fast byte-stream Run-Length and zero-sequence encoder for screen tiles.

/// Compresses a 4096-byte tile buffer using zero-run length encoding.
///
/// Format:
/// - A byte with MSB=1 indicates a run of zeroes: `0x80 | (count - 1)` (runs from 1 to 128 zeroes).
/// - A byte with MSB=0 indicates literal length: `(lit_count - 1)` followed by literal bytes.
pub fn compress_tile_rle(input: &[u8], output: &mut Vec<u8>) {
    let mut i = 0;
    let len = input.len();

    while i < len {
        // Check for run of zeroes
        if input[i] == 0 {
            let start = i;
            while i < len && input[i] == 0 && (i - start) < 128 {
                i += 1;
            }
            let zero_count = (i - start) as u8;
            output.push(0x80 | (zero_count - 1));
        } else {
            // Literal sequence
            let start = i;
            while i < len
                && (input[i] != 0 || (i + 1 < len && input[i + 1] != 0))
                && (i - start) < 128
            {
                i += 1;
            }
            let lit_count = (i - start) as u8;
            output.push(lit_count - 1);
            output.extend_from_slice(&input[start..i]);
        }
    }
}

/// Decompresses zero-run encoded data into an output buffer.
pub fn decompress_tile_rle(compressed: &[u8], output: &mut [u8]) -> Result<usize, &'static str> {
    let mut in_pos = 0;
    let mut out_pos = 0;
    let out_len = output.len();

    while in_pos < compressed.len() {
        let tag = compressed[in_pos];
        in_pos += 1;

        if (tag & 0x80) != 0 {
            // Zero run
            let count = ((tag & 0x7F) + 1) as usize;
            if out_pos + count > out_len {
                return Err("Output buffer overflow during zero-run decompression");
            }
            output[out_pos..out_pos + count].fill(0);
            out_pos += count;
        } else {
            // Literal run
            let count = (tag + 1) as usize;
            if in_pos + count > compressed.len() {
                return Err("Truncated literal run in compressed stream");
            }
            if out_pos + count > out_len {
                return Err("Output buffer overflow during literal decompression");
            }
            output[out_pos..out_pos + count].copy_from_slice(&compressed[in_pos..in_pos + count]);
            in_pos += count;
            out_pos += count;
        }
    }

    Ok(out_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_rle_roundtrip() {
        let mut original = vec![0u8; 4096];
        // Insert some non-zero patches
        original[10..20].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        original[500..505].copy_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC, 0xFB]);

        let mut compressed = Vec::new();
        compress_tile_rle(&original, &mut compressed);

        // Highly compressible (mostly zeroes)
        assert!(compressed.len() < 100);

        let mut decompressed = vec![0u8; 4096];
        let bytes_written =
            decompress_tile_rle(&compressed, &mut decompressed).expect("Decompress successfully");
        assert_eq!(bytes_written, 4096);
        assert_eq!(original, decompressed);
    }
}
