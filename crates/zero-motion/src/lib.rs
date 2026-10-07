//! # ZeroMotion
//!
//! Robotics kinematics and trajectory generation suite for ZeroRust:
//! - 6-Axis, SCARA, and Cartesian Kinematics
//! - Forward Kinematics (FK) and Inverse Kinematics (IK) solvers
//! - 7-phase Jerk-limited S-curve motion profile generator with continuous sampling

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod kinematics;
pub mod scurve;
pub mod types;

pub use kinematics::{Articulated6Dof, DhParam, ScaraJoints, ScaraRobot};
pub use scurve::{SCurveConstraints, SCurvePlanner, SCurveTrajectory, TrajectoryPoint};
pub use types::{Point3D, Pose, Vector3D};
