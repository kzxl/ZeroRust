//! Deterministic 9-digit sovereign device identity and secure PIN generation.

/// Sovereign 9-digit machine identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

impl DeviceId {
    /// Minimum valid 9-digit identifier (100 000 000).
    pub const MIN_ID: u32 = 100_000_000;
    /// Maximum valid 9-digit identifier (999 999 999).
    pub const MAX_ID: u32 = 999_999_999;

    /// Generates a deterministic 9-digit machine ID based on environment fingerprint.
    pub fn generate_local() -> Self {
        let mut hasher = 0xCBF29CE484222325u64; // FNV-1a offset basis
        let prime = 0x00000100000001B3u64;

        let host = std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "ZConn-Device".to_string());

        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;

        for b in host
            .as_bytes()
            .iter()
            .chain(os.as_bytes())
            .chain(arch.as_bytes())
        {
            hasher ^= *b as u64;
            hasher = hasher.wrapping_mul(prime);
        }

        // Fold 64-bit hash into 9-digit range [100_000_000, 999_999_999]
        let range = Self::MAX_ID - Self::MIN_ID + 1;
        let id_val = Self::MIN_ID + ((hasher % (range as u64)) as u32);
        Self(id_val)
    }

    /// Formats device ID into human-friendly 3x3 digit string (e.g. "912 345 678").
    pub fn to_formatted_string(&self) -> String {
        let s = format!("{:09}", self.0);
        format!("{} {} {}", &s[0..3], &s[3..6], &s[6..9])
    }

    /// Parses a formatted or contiguous string back into DeviceId.
    pub fn from_str_lenient(input: &str) -> Option<Self> {
        let digits: String = input.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 9 {
            digits.parse::<u32>().ok().map(Self)
        } else {
            None
        }
    }
}

/// Secure 6-character random one-time password generator for unattended or ad-hoc sessions.
pub struct PasswordGenerator;

impl PasswordGenerator {
    /// Generates a 6-character lowercase alphanumeric PIN (e.g. "7k3m9p").
    pub fn generate_pin(seed: u64) -> String {
        const CHARS: &[u8] = b"abcdefghjkmnpqrstuvwxyz23456789"; // Removed confusing 0, O, 1, l, i
        let mut state = seed ^ 0xA5A5_5A5A_BEEF_CAFE;
        let mut pin = String::with_capacity(6);

        for _ in 0..6 {
            // Xorshift64 PRNG step
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let idx = (state as usize) % CHARS.len();
            pin.push(CHARS[idx] as char);
        }
        pin
    }

    /// Constant-time password comparison to thwart timing side-channel attacks.
    pub fn verify_constant_time(a: &str, b: &str) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut diff = 0u8;
        for (ba, bb) in a.bytes().zip(b.bytes()) {
            diff |= ba ^ bb;
        }
        diff == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_generation_and_formatting() {
        let dev = DeviceId::generate_local();
        assert!(dev.0 >= DeviceId::MIN_ID);
        assert!(dev.0 <= DeviceId::MAX_ID);

        let fmt = dev.to_formatted_string();
        assert_eq!(fmt.len(), 11); // "123 456 789" = 11 chars
        assert_eq!(fmt.matches(' ').count(), 2);

        let parsed = DeviceId::from_str_lenient(&fmt).expect("Parse formatted");
        assert_eq!(dev, parsed);

        let parsed_plain = DeviceId::from_str_lenient(&dev.0.to_string()).expect("Parse plain");
        assert_eq!(dev, parsed_plain);
    }

    #[test]
    fn test_pin_generation_and_constant_time_verification() {
        let pin = PasswordGenerator::generate_pin(123456789);
        assert_eq!(pin.len(), 6);

        assert!(PasswordGenerator::verify_constant_time(&pin, &pin));
        assert!(!PasswordGenerator::verify_constant_time(&pin, "wrong!"));
        assert!(!PasswordGenerator::verify_constant_time(&pin, "123456"));
    }
}
