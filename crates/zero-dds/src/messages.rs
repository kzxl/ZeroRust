//! Standard ROS 2 message definitions with zero-allocation CDR serializers.

use crate::cdr::{CdrReader, CdrWriter};
use zero_core::error::ZeroResult;

/// `builtin_interfaces/Time` message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Time {
    /// Seconds component of timestamp.
    pub sec: i32,
    /// Nanoseconds component of timestamp (0..999,999,999).
    pub nanosec: u32,
}

impl Time {
    /// Serializes Time into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        writer.write_i32(self.sec)?;
        writer.write_u32(self.nanosec)?;
        Ok(())
    }

    /// Deserializes Time from CDR stream.
    pub fn deserialize(reader: &mut CdrReader) -> ZeroResult<Self> {
        let sec = reader.read_i32()?;
        let nanosec = reader.read_u32()?;
        Ok(Self { sec, nanosec })
    }
}

/// `std_msgs/Header` message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header<'a> {
    /// Timestamp.
    pub stamp: Time,
    /// Coordinate frame identifier string.
    pub frame_id: &'a str,
}

impl<'a> Header<'a> {
    /// Serializes Header into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        self.stamp.serialize(writer)?;
        writer.write_str(self.frame_id)?;
        Ok(())
    }

    /// Deserializes Header from CDR stream.
    pub fn deserialize(reader: &mut CdrReader<'a>) -> ZeroResult<Self> {
        let stamp = Time::deserialize(reader)?;
        let frame_id = reader.read_str()?;
        Ok(Self { stamp, frame_id })
    }
}

/// `geometry_msgs/Vector3` message.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3 {
    /// X axis coordinate.
    pub x: f64,
    /// Y axis coordinate.
    pub y: f64,
    /// Z axis coordinate.
    pub z: f64,
}

impl Vector3 {
    /// Creates a new Vector3.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Serializes Vector3 into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        writer.write_f64(self.x)?;
        writer.write_f64(self.y)?;
        writer.write_f64(self.z)?;
        Ok(())
    }

    /// Deserializes Vector3 from CDR stream.
    pub fn deserialize(reader: &mut CdrReader) -> ZeroResult<Self> {
        let x = reader.read_f64()?;
        let y = reader.read_f64()?;
        let z = reader.read_f64()?;
        Ok(Self { x, y, z })
    }
}

/// `geometry_msgs/Quaternion` message.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quaternion {
    /// Quaternion x component.
    pub x: f64,
    /// Quaternion y component.
    pub y: f64,
    /// Quaternion z component.
    pub z: f64,
    /// Quaternion w scalar component.
    pub w: f64,
}

impl Quaternion {
    /// Creates an identity quaternion [0, 0, 0, 1].
    pub const fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    /// Serializes Quaternion into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        writer.write_f64(self.x)?;
        writer.write_f64(self.y)?;
        writer.write_f64(self.z)?;
        writer.write_f64(self.w)?;
        Ok(())
    }

    /// Deserializes Quaternion from CDR stream.
    pub fn deserialize(reader: &mut CdrReader) -> ZeroResult<Self> {
        let x = reader.read_f64()?;
        let y = reader.read_f64()?;
        let z = reader.read_f64()?;
        let w = reader.read_f64()?;
        Ok(Self { x, y, z, w })
    }
}

/// `geometry_msgs/Twist` message for linear and angular mobile base velocity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Twist {
    /// Linear velocity (m/s).
    pub linear: Vector3,
    /// Angular velocity (rad/s).
    pub angular: Vector3,
}

impl Twist {
    /// Serializes Twist into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        self.linear.serialize(writer)?;
        self.angular.serialize(writer)?;
        Ok(())
    }

    /// Deserializes Twist from CDR stream.
    pub fn deserialize(reader: &mut CdrReader) -> ZeroResult<Self> {
        let linear = Vector3::deserialize(reader)?;
        let angular = Vector3::deserialize(reader)?;
        Ok(Self { linear, angular })
    }
}

/// `sensor_msgs/Imu` message for 6-axis / 9-axis inertial measurement units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imu<'a> {
    /// Standard ROS 2 header.
    pub header: Header<'a>,
    /// Orientation quaternion.
    pub orientation: Quaternion,
    /// Angular velocity from gyroscopes (rad/s).
    pub angular_velocity: Vector3,
    /// Linear acceleration from accelerometers (m/s^2).
    pub linear_acceleration: Vector3,
}

impl<'a> Imu<'a> {
    /// Serializes IMU message into CDR stream.
    pub fn serialize(&self, writer: &mut CdrWriter) -> ZeroResult<()> {
        self.header.serialize(writer)?;
        self.orientation.serialize(writer)?;
        // 9-element orientation covariance (omitted as zeros for compact serialization)
        for _ in 0..9 {
            writer.write_f64(0.0)?;
        }
        self.angular_velocity.serialize(writer)?;
        for _ in 0..9 {
            writer.write_f64(0.0)?;
        }
        self.linear_acceleration.serialize(writer)?;
        for _ in 0..9 {
            writer.write_f64(0.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twist_cdr_roundtrip() {
        let twist = Twist {
            linear: Vector3::new(1.5, -0.2, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.85),
        };

        let mut buf = [0u8; 128];
        let mut writer = CdrWriter::new(&mut buf);
        twist.serialize(&mut writer).unwrap();

        let len = writer.position();
        let mut reader = CdrReader::new(&buf[..len]);
        let parsed = Twist::deserialize(&mut reader).unwrap();

        assert!((parsed.linear.x - 1.5).abs() < 1e-6);
        assert!((parsed.linear.y - (-0.2)).abs() < 1e-6);
        assert!((parsed.angular.z - 0.85).abs() < 1e-6);
    }
}
