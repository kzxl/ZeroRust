//! # ZeroDynamics
//!
//! High-performance Robotics Dynamics and Collaborative Control suite for ZeroRust:
//! - $O(N)$ Recursive Newton-Euler Algorithm (RNEA) inverse dynamics solver
//! - Exact Mass Matrix $M(q)$, Gravity vector $g(q)$, and Coriolis terms $C(q, \dot{q})\dot{q}$
//! - Geometric Jacobian and Yoshikawa Manipulability measure
//! - Sensorless Collision Detection via generalized momentum observer
//! - Cartesian Impedance and Admittance controllers for safe human-robot collaboration

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]

pub mod collision;
pub mod impedance;
pub mod jacobian;
pub mod link;
pub mod rnea;
pub mod spatial;

pub use collision::{CollisionStatus, MomentumObserver};
pub use impedance::{AdmittanceController, Cartesian6D, ImpedanceController};
pub use jacobian::GeometricJacobian;
pub use link::{InertiaTensor, Link};
pub use rnea::RneaSolver;
pub use spatial::{Mat3, Vec3};
