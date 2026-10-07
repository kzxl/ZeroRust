//! Physical USB HID Usage Code (Page 0x07) definitions and cross-platform key mapping.

/// Standard USB HID Keyboard Usage Codes (Page 0x07).
pub mod hid {
    /// Key 'A' (physical key position 1 on home row).
    pub const KEY_A: u8 = 0x04;
    /// Key 'B'.
    pub const KEY_B: u8 = 0x05;
    /// Key 'C'.
    pub const KEY_C: u8 = 0x06;
    /// Key 'D'.
    pub const KEY_D: u8 = 0x07;
    /// Key 'E'.
    pub const KEY_E: u8 = 0x08;
    /// Key 'F'.
    pub const KEY_F: u8 = 0x09;
    /// Key 'G'.
    pub const KEY_G: u8 = 0x0A;
    /// Key 'H'.
    pub const KEY_H: u8 = 0x0B;
    /// Key 'I'.
    pub const KEY_I: u8 = 0x0C;
    /// Key 'J'.
    pub const KEY_J: u8 = 0x0D;
    /// Key 'K'.
    pub const KEY_K: u8 = 0x0E;
    /// Key 'L'.
    pub const KEY_L: u8 = 0x0F;
    /// Key 'M'.
    pub const KEY_M: u8 = 0x10;
    /// Key 'N'.
    pub const KEY_N: u8 = 0x11;
    /// Key 'O'.
    pub const KEY_O: u8 = 0x12;
    /// Key 'P'.
    pub const KEY_P: u8 = 0x13;
    /// Key 'Q'.
    pub const KEY_Q: u8 = 0x14;
    /// Key 'R'.
    pub const KEY_R: u8 = 0x15;
    /// Key 'S'.
    pub const KEY_S: u8 = 0x16;
    /// Key 'T'.
    pub const KEY_T: u8 = 0x17;
    /// Key 'U'.
    pub const KEY_U: u8 = 0x18;
    /// Key 'V'.
    pub const KEY_V: u8 = 0x19;
    /// Key 'W'.
    pub const KEY_W: u8 = 0x1A;
    /// Key 'X'.
    pub const KEY_X: u8 = 0x1B;
    /// Key 'Y'.
    pub const KEY_Y: u8 = 0x1C;
    /// Key 'Z'.
    pub const KEY_Z: u8 = 0x1D;

    /// Number row '1' and '!'.
    pub const KEY_1: u8 = 0x1E;
    /// Number row '2' and '@'.
    pub const KEY_2: u8 = 0x1F;
    /// Number row '3' and '#'.
    pub const KEY_3: u8 = 0x20;
    /// Number row '4' and '$'.
    pub const KEY_4: u8 = 0x21;
    /// Number row '5' and '%'.
    pub const KEY_5: u8 = 0x22;
    /// Number row '6' and '^'.
    pub const KEY_6: u8 = 0x23;
    /// Number row '7' and '&'.
    pub const KEY_7: u8 = 0x24;
    /// Number row '8' and '*'.
    pub const KEY_8: u8 = 0x25;
    /// Number row '9' and '('.
    pub const KEY_9: u8 = 0x26;
    /// Number row '0' and ')'.
    pub const KEY_0: u8 = 0x27;

    /// Return / Enter key.
    pub const KEY_ENTER: u8 = 0x28;
    /// Escape key.
    pub const KEY_ESCAPE: u8 = 0x29;
    /// Backspace key.
    pub const KEY_BACKSPACE: u8 = 0x2A;
    /// Tab key.
    pub const KEY_TAB: u8 = 0x2B;
    /// Spacebar key.
    pub const KEY_SPACE: u8 = 0x2C;

    /// Left Control modifier.
    pub const KEY_LEFT_CTRL: u8 = 0xE0;
    /// Left Shift modifier.
    pub const KEY_LEFT_SHIFT: u8 = 0xE1;
    /// Left Alt / Option modifier.
    pub const KEY_LEFT_ALT: u8 = 0xE2;
    /// Left GUI / Windows / Command modifier.
    pub const KEY_LEFT_GUI: u8 = 0xE3;
    /// Right Control modifier.
    pub const KEY_RIGHT_CTRL: u8 = 0xE4;
    /// Right Shift modifier.
    pub const KEY_RIGHT_SHIFT: u8 = 0xE5;
    /// Right Alt / Option modifier.
    pub const KEY_RIGHT_ALT: u8 = 0xE6;
    /// Right GUI / Windows / Command modifier.
    pub const KEY_RIGHT_GUI: u8 = 0xE7;
}

/// Translates USB HID scancode into Windows Virtual-Key code (VK).
pub fn hid_to_windows_vk(hid_code: u8) -> u16 {
    match hid_code {
        hid::KEY_A..=hid::KEY_Z => 0x41 + (hid_code - hid::KEY_A) as u16,
        hid::KEY_1..=hid::KEY_9 => 0x31 + (hid_code - hid::KEY_1) as u16,
        hid::KEY_0 => 0x30,
        hid::KEY_ENTER => 0x0D,       // VK_RETURN
        hid::KEY_ESCAPE => 0x1B,      // VK_ESCAPE
        hid::KEY_BACKSPACE => 0x08,   // VK_BACK
        hid::KEY_TAB => 0x09,         // VK_TAB
        hid::KEY_SPACE => 0x20,       // VK_SPACE
        hid::KEY_LEFT_CTRL => 0xA2,   // VK_LCONTROL
        hid::KEY_RIGHT_CTRL => 0xA3,  // VK_RCONTROL
        hid::KEY_LEFT_SHIFT => 0xA0,  // VK_LSHIFT
        hid::KEY_RIGHT_SHIFT => 0xA1, // VK_RSHIFT
        hid::KEY_LEFT_ALT => 0xA4,    // VK_LMENU
        hid::KEY_RIGHT_ALT => 0xA5,   // VK_RMENU
        hid::KEY_LEFT_GUI => 0x5B,    // VK_LWIN
        hid::KEY_RIGHT_GUI => 0x5C,   // VK_RWIN
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hid_mapping() {
        assert_eq!(hid_to_windows_vk(hid::KEY_A), 0x41); // 'A'
        assert_eq!(hid_to_windows_vk(hid::KEY_ENTER), 0x0D);
        assert_eq!(hid_to_windows_vk(hid::KEY_LEFT_CTRL), 0xA2);
    }
}
