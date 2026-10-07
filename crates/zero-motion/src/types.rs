//! Geometry and coordinate types for robotics and spatial motion.

/// A 3D coordinate point in millimeters or meters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Creates a new `Point3D`.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Computes Euclidean distance to another point.
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        #[cfg(feature = "std")]
        {
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            // Simple newton sqrt if no_std
            let val = dx * dx + dy * dy + dz * dz;
            if val <= 0.0 {
                0.0
            } else {
                libm::sqrt(val)
            }
        }
    }
}

/// A 3D spatial vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3D {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vector3D {
    /// Creates a new `Vector3D`.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Dot product between two vectors.
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product between two vectors.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

/// Spatial 6-DOF Pose representing position and Euler orientation (Roll, Pitch, Yaw).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pose {
    /// Spatial position.
    pub position: Point3D,
    /// Rotation about X axis (radians).
    pub roll: f64,
    /// Rotation about Y axis (radians).
    pub pitch: f64,
    /// Rotation about Z axis (radians).
    pub yaw: f64,
}

impl Pose {
    /// Creates a new `Pose`.
    pub const fn new(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            position: Point3D::new(x, y, z),
            roll,
            pitch,
            yaw,
        }
    }
}
