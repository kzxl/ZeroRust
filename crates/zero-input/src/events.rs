//! Input event definitions and binary wire encoding.

/// State of an interactive button or key.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementState {
    /// Button or key is released (up).
    Released = 0,
    /// Button or key is pressed (down).
    Pressed = 1,
}

impl ElementState {
    /// Decodes element state from wire byte.
    pub fn from_u8(b: u8) -> Self {
        if b == 1 {
            Self::Pressed
        } else {
            Self::Released
        }
    }
}

/// Mouse button identifiers.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    /// Primary left mouse button.
    Left = 0,
    /// Secondary right mouse button.
    Right = 1,
    /// Middle mouse wheel button.
    Middle = 2,
    /// Back auxiliary button (mouse button 4).
    Back = 3,
    /// Forward auxiliary button (mouse button 5).
    Forward = 4,
}

impl MouseButton {
    /// Decodes mouse button from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Left),
            1 => Some(Self::Right),
            2 => Some(Self::Middle),
            3 => Some(Self::Back),
            4 => Some(Self::Forward),
            _ => None,
        }
    }
}

/// Mouse interaction events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseEvent {
    /// Absolute cursor motion with normalized coordinates ($[0.0, 1.0]$).
    MoveAbsolute {
        /// Normalized horizontal coordinate (0.0 = left edge, 1.0 = right edge).
        norm_x: f32,
        /// Normalized vertical coordinate (0.0 = top edge, 1.0 = bottom edge).
        norm_y: f32,
        /// Target display monitor index.
        display_id: u8,
    },
    /// Relative cursor delta motion (for FPS / 3D CAD camera orbit).
    MoveRelative {
        /// Horizontal pixel motion delta.
        delta_x: i32,
        /// Vertical pixel motion delta.
        delta_y: i32,
    },
    /// Mouse button press or release.
    Button {
        /// Target mouse button.
        button: MouseButton,
        /// New pressed or released state.
        state: ElementState,
    },
    /// Mouse wheel rotation.
    Wheel {
        /// Horizontal scroll wheel delta.
        delta_x: i16,
        /// Vertical scroll wheel delta.
        delta_y: i16,
    },
}

impl MouseEvent {
    /// Serializes mouse event into a binary byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        match *self {
            Self::MoveAbsolute {
                norm_x,
                norm_y,
                display_id,
            } => {
                b.push(0x01); // Tag
                b.extend_from_slice(&norm_x.to_le_bytes());
                b.extend_from_slice(&norm_y.to_le_bytes());
                b.push(display_id);
            }
            Self::MoveRelative { delta_x, delta_y } => {
                b.push(0x02); // Tag
                b.extend_from_slice(&delta_x.to_le_bytes());
                b.extend_from_slice(&delta_y.to_le_bytes());
            }
            Self::Button { button, state } => {
                b.push(0x03); // Tag
                b.push(button as u8);
                b.push(state as u8);
            }
            Self::Wheel { delta_x, delta_y } => {
                b.push(0x04); // Tag
                b.extend_from_slice(&delta_x.to_le_bytes());
                b.extend_from_slice(&delta_y.to_le_bytes());
            }
        }
        b
    }

    /// Deserializes mouse event from a byte slice.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.is_empty() {
            return None;
        }
        match slice[0] {
            0x01 => {
                if slice.len() < 10 {
                    return None;
                }
                let norm_x = f32::from_le_bytes([slice[1], slice[2], slice[3], slice[4]]);
                let norm_y = f32::from_le_bytes([slice[5], slice[6], slice[7], slice[8]]);
                let display_id = slice[9];
                Some(Self::MoveAbsolute {
                    norm_x,
                    norm_y,
                    display_id,
                })
            }
            0x02 => {
                if slice.len() < 9 {
                    return None;
                }
                let delta_x = i32::from_le_bytes([slice[1], slice[2], slice[3], slice[4]]);
                let delta_y = i32::from_le_bytes([slice[5], slice[6], slice[7], slice[8]]);
                Some(Self::MoveRelative { delta_x, delta_y })
            }
            0x03 => {
                if slice.len() < 3 {
                    return None;
                }
                let button = MouseButton::from_u8(slice[1])?;
                let state = ElementState::from_u8(slice[2]);
                Some(Self::Button { button, state })
            }
            0x04 => {
                if slice.len() < 5 {
                    return None;
                }
                let delta_x = i16::from_le_bytes([slice[1], slice[2]]);
                let delta_y = i16::from_le_bytes([slice[3], slice[4]]);
                Some(Self::Wheel { delta_x, delta_y })
            }
            _ => None,
        }
    }
}

/// Keyboard interaction event using USB HID physical scancodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyboardEvent {
    /// USB HID Usage Table (Page 0x07) scancode.
    pub scancode: u8,
    /// Press or release state.
    pub state: ElementState,
    /// Modifier bitmask: 0x01=Shift, 0x02=Ctrl, 0x04=Alt, 0x08=Meta/Win.
    pub modifiers: u8,
}

impl KeyboardEvent {
    /// Serializes keyboard event into binary bytes (3 bytes).
    pub fn to_bytes(&self) -> [u8; 3] {
        [self.scancode, self.state as u8, self.modifiers]
    }

    /// Deserializes keyboard event from binary bytes.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < 3 {
            return None;
        }
        Some(Self {
            scancode: slice[0],
            state: ElementState::from_u8(slice[1]),
            modifiers: slice[2],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mouse_event_serialization() {
        let ev = MouseEvent::MoveAbsolute {
            norm_x: 0.5,
            norm_y: 0.75,
            display_id: 1,
        };
        let b = ev.to_bytes();
        let parsed = MouseEvent::from_bytes(&b).expect("Parse mouse event");
        assert_eq!(ev, parsed);

        let btn = MouseEvent::Button {
            button: MouseButton::Right,
            state: ElementState::Pressed,
        };
        let b_btn = btn.to_bytes();
        let parsed_btn = MouseEvent::from_bytes(&b_btn).expect("Parse button");
        assert_eq!(btn, parsed_btn);
    }

    #[test]
    fn test_keyboard_event_serialization() {
        let k = KeyboardEvent {
            scancode: 0x04, // 'A'
            state: ElementState::Pressed,
            modifiers: 0x02, // Ctrl
        };
        let b = k.to_bytes();
        let parsed = KeyboardEvent::from_bytes(&b).expect("Parse keyboard");
        assert_eq!(k, parsed);
    }
}
