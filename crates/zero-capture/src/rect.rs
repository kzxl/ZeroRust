//! Screen geometry, display information, and bounding rectangles.

/// Bounding rectangle in screen coordinate space.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// Left horizontal offset in pixels.
    pub x: u32,
    /// Top vertical offset in pixels.
    pub y: u32,
    /// Width of the rectangle in pixels.
    pub width: u32,
    /// Height of the rectangle in pixels.
    pub height: u32,
}

impl Rect {
    /// Creates a new bounding rectangle.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Computes the pixel area of the rectangle.
    pub const fn area(&self) -> u64 {
        (self.width as u64) * (self.height as u64)
    }

    /// Checks if a coordinate is contained within the rectangle.
    pub const fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < (self.x + self.width) && py >= self.y && py < (self.y + self.height)
    }

    /// Computes the bounding box union of two rectangles.
    pub fn union(&self, other: &Rect) -> Rect {
        if self.width == 0 || self.height == 0 {
            return *other;
        }
        if other.width == 0 || other.height == 0 {
            return *self;
        }

        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);

        Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

/// Metadata and physical properties of a connected display monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayInfo {
    /// Display monitor index (0-based).
    pub id: usize,
    /// Friendly human-readable monitor name (e.g. "Primary Display", "HDMI-1").
    pub name: String,
    /// Resolution width in physical pixels.
    pub width: u32,
    /// Resolution height in physical pixels.
    pub height: u32,
    /// Virtual desktop horizontal offset.
    pub offset_x: i32,
    /// Virtual desktop vertical offset.
    pub offset_y: i32,
    /// Refresh rate in Hertz (e.g. 60, 120, 144).
    pub refresh_rate_hz: u32,
    /// Flag indicating whether this is the primary desktop display.
    pub is_primary: bool,
    /// DPI scaling percentage (100 = 1.0x, 150 = 1.5x, 200 = 2.0x).
    pub scale_factor_pct: u32,
}

impl Default for DisplayInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Primary Display".to_string(),
            width: 1920,
            height: 1080,
            offset_x: 0,
            offset_y: 0,
            refresh_rate_hz: 60,
            is_primary: true,
            scale_factor_pct: 100,
        }
    }
}

/// Multi-monitor virtual desktop topology description.
#[derive(Debug, Clone, Default)]
pub struct DisplayTopology {
    /// List of active display outputs.
    pub displays: Vec<DisplayInfo>,
    /// Combined virtual desktop width encompassing all displays.
    pub virtual_width: u32,
    /// Combined virtual desktop height encompassing all displays.
    pub virtual_height: u32,
}

impl DisplayTopology {
    /// Creates a single-monitor default topology with 1080p resolution.
    pub fn standard_1080p() -> Self {
        Self {
            displays: vec![DisplayInfo::default()],
            virtual_width: 1920,
            virtual_height: 1080,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_operations() {
        let r1 = Rect::new(0, 0, 100, 100);
        let r2 = Rect::new(50, 50, 100, 100);
        assert_eq!(r1.area(), 10000);
        assert!(r1.contains(50, 50));
        assert!(!r1.contains(100, 100));

        let u = r1.union(&r2);
        assert_eq!(u.x, 0);
        assert_eq!(u.y, 0);
        assert_eq!(u.width, 150);
        assert_eq!(u.height, 150);
    }
}
