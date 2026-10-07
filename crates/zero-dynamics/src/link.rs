//! Rigid body link inertia and kinematic joint parameters.

use crate::spatial::{Mat3, Vec3};

/// 3x3 Inertia tensor represented at the link's center of mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertiaTensor {
    /// 3x3 rotational inertia matrix [kg*m^2].
    pub mat: Mat3,
}

impl Default for InertiaTensor {
    fn default() -> Self {
        Self { mat: Mat3::zero() }
    }
}

impl InertiaTensor {
    /// Creates a principal diagonal inertia tensor: diag(Ixx, Iyy, Izz).
    pub const fn principal(ixx: f64, iyy: f64, izz: f64) -> Self {
        Self {
            mat: Mat3 {
                data: [[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]],
            },
        }
    }

    /// Creates a full symmetric inertia tensor.
    pub const fn new(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64) -> Self {
        Self {
            mat: Mat3 {
                data: [[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]],
            },
        }
    }
}

/// Dynamic and kinematic link model for an individual robotic arm segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// DH link length $a_i$ (meters).
    pub a: f64,
    /// DH link twist $\alpha_i$ (radians).
    pub alpha: f64,
    /// DH link offset $d_i$ (meters).
    pub d: f64,
    /// DH joint offset angle $\theta_0$ (radians).
    pub theta_offset: f64,
    /// Mass of the link $m_i$ (kg).
    pub mass: f64,
    /// Center of mass vector $\mathbf{r}_c$ relative to link $i$ coordinate frame (meters).
    pub com: Vec3,
    /// 3x3 Inertia tensor about center of mass.
    pub inertia: InertiaTensor,
    /// Joint viscous damping coefficient $b_i$ [N*m*s/rad].
    pub damping: f64,
    /// Joint static Coulomb friction $c_i$ [N*m].
    pub coulomb_friction: f64,
}

impl Default for Link {
    fn default() -> Self {
        Self {
            a: 0.0,
            alpha: 0.0,
            d: 0.0,
            theta_offset: 0.0,
            mass: 0.0,
            com: Vec3::zero(),
            inertia: InertiaTensor::default(),
            damping: 0.0,
            coulomb_friction: 0.0,
        }
    }
}

impl Link {
    /// Computes the link offset vector $\mathbf{p}_i = [a_i, d_i \sin\alpha_i, d_i \cos\alpha_i]^T$
    /// connecting origin of frame $i-1$ to frame $i$ in frame $i$ coordinates.
    pub fn link_offset_vector(&self) -> Vec3 {
        #[cfg(feature = "std")]
        let (sa, ca) = (self.alpha.sin(), self.alpha.cos());
        #[cfg(not(feature = "std"))]
        let (sa, ca) = (libm::sin(self.alpha), libm::cos(self.alpha));

        Vec3::new(self.a, self.d * sa, self.d * ca)
    }

    /// Friction torque for given joint velocity $\dot{q}_i$:
    /// $\tau_f = b_i \dot{q}_i + c_i \text{sign}(\dot{q}_i)$
    pub fn friction_torque(&self, q_dot: f64) -> f64 {
        let sign = if q_dot > 1e-6 {
            1.0
        } else if q_dot < -1e-6 {
            -1.0
        } else {
            0.0
        };
        self.damping * q_dot + self.coulomb_friction * sign
    }
}
