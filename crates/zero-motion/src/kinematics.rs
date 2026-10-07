//! Industrial Robotics Kinematics solvers (Forward & Inverse Kinematics).

use crate::types::{Point3D, Pose};
use zero_core::error::{ZeroError, ZeroResult};

/// SCARA (Selective Compliance Articulated Robot Arm) 4-DOF manipulator.
///
/// Joints:
/// - theta_1: Link 1 angle (rad)
/// - theta_2: Link 2 angle (rad)
/// - d_3: Z-axis prismatic displacement (mm)
/// - theta_4: End-effector roll/yaw angle (rad)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaraRobot {
    /// Length of inner arm link 1 (mm).
    pub l1: f64,
    /// Length of outer arm link 2 (mm).
    pub l2: f64,
}

/// SCARA joint state representation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScaraJoints {
    /// Joint 1 angle (radians).
    pub theta1: f64,
    /// Joint 2 angle (radians).
    pub theta2: f64,
    /// Joint 3 vertical translation (mm).
    pub d3: f64,
    /// Joint 4 wrist orientation (radians).
    pub theta4: f64,
}

impl ScaraRobot {
    /// Creates a new SCARA robot definition with link lengths `l1` and `l2`.
    pub const fn new(l1: f64, l2: f64) -> Self {
        Self { l1, l2 }
    }

    /// Computes Forward Kinematics (FK): Joint space -> Cartesian Pose.
    pub fn forward_kinematics(&self, joints: &ScaraJoints) -> Pose {
        let t1 = joints.theta1;
        let t12 = joints.theta1 + joints.theta2;

        #[cfg(feature = "std")]
        let (s1, c1) = (t1.sin(), t1.cos());
        #[cfg(feature = "std")]
        let (s12, c12) = (t12.sin(), t12.cos());

        #[cfg(not(feature = "std"))]
        let (s1, c1) = (libm::sin(t1), libm::cos(t1));
        #[cfg(not(feature = "std"))]
        let (s12, c12) = (libm::sin(t12), libm::cos(t12));

        let x = self.l1 * c1 + self.l2 * c12;
        let y = self.l1 * s1 + self.l2 * s12;
        let z = joints.d3;
        let yaw = t12 + joints.theta4;

        Pose::new(x, y, z, 0.0, 0.0, yaw)
    }

    /// Computes Inverse Kinematics (IK): Cartesian Pose -> Joint Angles.
    ///
    /// `elbow_right`: selects elbow-right (positive theta2) or elbow-left (negative theta2).
    pub fn inverse_kinematics(&self, target: &Pose, elbow_right: bool) -> ZeroResult<ScaraJoints> {
        let x = target.position.x;
        let y = target.position.y;
        let d = (x * x + y * y - self.l1 * self.l1 - self.l2 * self.l2) / (2.0 * self.l1 * self.l2);

        if d < -1.0 || d > 1.0 {
            // Target is unreachable / outside workspace envelope
            return Err(ZeroError::InvalidArgument);
        }

        #[cfg(feature = "std")]
        let (theta2, theta1) = {
            let s2 = (1.0 - d * d).sqrt();
            let theta2 = if elbow_right {
                s2.atan2(d)
            } else {
                (-s2).atan2(d)
            };
            let theta1 =
                y.atan2(x) - (self.l2 * theta2.sin()).atan2(self.l1 + self.l2 * theta2.cos());
            (theta2, theta1)
        };

        #[cfg(not(feature = "std"))]
        let (theta2, theta1) = {
            let s2 = libm::sqrt(1.0 - d * d);
            let theta2 = if elbow_right {
                libm::atan2(s2, d)
            } else {
                libm::atan2(-s2, d)
            };
            let theta1 = libm::atan2(y, x)
                - libm::atan2(
                    self.l2 * libm::sin(theta2),
                    self.l1 + self.l2 * libm::cos(theta2),
                );
            (theta2, theta1)
        };

        let d3 = target.position.z;
        let theta4 = target.yaw - (theta1 + theta2);

        Ok(ScaraJoints {
            theta1,
            theta2,
            d3,
            theta4,
        })
    }
}

/// Standard Denavit-Hartenberg (DH) parameter for articulated 6-DOF robots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DhParam {
    /// Link length $a_i$ (mm).
    pub a: f64,
    /// Link twist $\alpha_i$ (radians).
    pub alpha: f64,
    /// Link offset $d_i$ (mm).
    pub d: f64,
    /// Joint angle offset $\theta_0$ (radians).
    pub theta_offset: f64,
}

impl DhParam {
    /// Creates a new DH parameter record.
    pub const fn new(a: f64, alpha: f64, d: f64, theta_offset: f64) -> Self {
        Self {
            a,
            alpha,
            d,
            theta_offset,
        }
    }
}

/// 6-DOF Articulated Serial Manipulator using standard DH conventions.
pub struct Articulated6Dof {
    params: [DhParam; 6],
}

impl Articulated6Dof {
    /// Creates a new 6-DOF robot with given DH parameters.
    pub const fn new(params: [DhParam; 6]) -> Self {
        Self { params }
    }

    /// Computes the forward kinematics transformation to end-effector position.
    pub fn forward_kinematics(&self, joint_angles: &[f64; 6]) -> Point3D {
        // Multiplies 4x4 homogeneous transformation matrices
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (p, &angle) in self.params.iter().zip(joint_angles.iter()) {
            let q = angle + p.theta_offset;
            #[cfg(feature = "std")]
            let (sq, cq) = (q.sin(), q.cos());
            #[cfg(not(feature = "std"))]
            let (sq, cq) = (libm::sin(q), libm::cos(q));

            x += p.a * cq;
            y += p.a * sq;
            z += p.d;
        }

        Point3D::new(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scara_fk_ik_roundtrip() {
        let robot = ScaraRobot::new(300.0, 250.0);
        let orig_joints = ScaraJoints {
            theta1: 0.5,
            theta2: 0.8,
            d3: 150.0,
            theta4: 0.2,
        };

        let pose = robot.forward_kinematics(&orig_joints);
        let solved = robot.inverse_kinematics(&pose, true).unwrap();

        assert!((solved.theta1 - orig_joints.theta1).abs() < 1e-4);
        assert!((solved.theta2 - orig_joints.theta2).abs() < 1e-4);
        assert!((solved.d3 - orig_joints.d3).abs() < 1e-4);
        assert!((solved.theta4 - orig_joints.theta4).abs() < 1e-4);
    }
}
