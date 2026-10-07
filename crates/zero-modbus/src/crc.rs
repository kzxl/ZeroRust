//! Modbus CRC16 calculation.

/// Computes the standard Modbus CRC16 checksum over a byte slice.
///
/// Polynomial: 0xA001 (reversed 0x8005), Initial Value: 0xFFFF.
pub fn calculate_crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;

    for &byte in data {
        crc ^= byte as u16;
        for _ in 0..8 {
            if (crc & 0x0001) != 0 {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }

    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modbus_crc16_known_vector() {
        // Expected CRC bytes: [0xC5, 0xCD] (Low byte 0xC5, High byte 0xCD -> 0xCDC5)
        let frame = [0x01, 0x03, 0x00, 0x00, 0x00, 0x0A];
        let crc = calculate_crc16(&frame);
        assert_eq!(crc, 0xCDC5);
        assert_eq!(crc.to_le_bytes(), [0xC5, 0xCD]);
    }
}
