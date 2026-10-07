//! Cartesian Impedance and Admittance controllers for compliant cobot interaction and hand-guiding.

use crate::jacobian::GeometricJacobian;
use crate::link::Link;
use crate::rnea::RneaSolver;

/// 6-DOF Cartesian parameter vector [X, Y, Z, Roll, Pitch, Yaw].
pub type Cartesian6D = [f64; 6];

/// Cartesian Impedance Controller defining virtual stiffness, damping, and inertia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpedanceController {
    /// Virtual stiffness coefficients $K_p$ [N/m or N*m/rad].
    pub stiffness: Cartesian6D,
    /// Virtual damping coefficients $D_p$ [N*s/m or N*m*s/rad].
    pub damping: Cartesian6D,
    /// Virtual inertia coefficients $M_d$ [kg or kg*m^2].
    pub inertia: Cartesian6D,
}

impl Default for ImpedanceController {
    fn default() -> Self {
        Self {
            stiffness: [500.0, 500.0, 500.0, 50.0, 50.0, 50.0],
            damping: [40.0, 40.0, 40.0, 5.0, 5.0, 5.0],
            inertia: [2.0, 2.0, 2.0, 0.2, 0.2, 0.2],
        }
    }
}

impl ImpedanceController {
    /// Creates a new impedance parameter set.
    pub const fn new(stiffness: Cartesian6D, damping: Cartesian6D, inertia: Cartesian6D) -> Self {
        Self {
            stiffness,
            damping,
            inertia,
        }
    }

    /// Computes the required Cartesian wrench $\mathbf{F}_{task}$ to enforce virtual dynamics:
    /// $\mathbf{F}_{task} = \mathbf{K}_p (\mathbf{x}_d - \mathbf{x}) + \mathbf{D}_p (\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + \mathbf{M}_d (\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}})$
    pub fn compute_cartesian_wrench(
        &self,
        x_desired: &Cartesian6D,
        x_actual: &Cartesian6D,
        xd_desired: &Cartesian6D,
        xd_actual: &Cartesian6D,
        xdd_desired: &Cartesian6D,
        xdd_actual: &Cartesian6D,
    ) -> Cartesian6D {
        let mut wrench = [0.0; 6];
        for i in 0..6 {
            let pos_err = x_desired[i] - x_actual[i];
            let vel_err = xd_desired[i] - xd_actual[i];
            let acc_err = xdd_desired[i] - xdd_actual[i];

            wrench[i] =
                self.stiffness[i] * pos_err + self.damping[i] * vel_err + self.inertia[i] * acc_err;
        }
        wrench
    }

    /// Computes compliant joint torques $\tau_{cmd} = \mathbf{J}^T \mathbf{F}_{task} + g(q)$.
    pub fn compute_joint_torques<const N: usize>(
        &self,
        jacobian: &GeometricJacobian<N>,
        solver: &mut RneaSolver<N>,
        links: &[Link; N],
        q: &[f64; N],
        task_wrench: &Cartesian6D,
        gravity: crate::spatial::Vec3,
    ) -> [f64; N] {
        let mut torques = jacobian.transpose_wrench(task_wrench);
        let g = solver.compute_gravity(links, q, gravity);
        for i in 0..N {
            torques[i] += g[i];
        }
        torques
    }
}

/// Admittance Controller for hand-guiding and lead-through teaching.
///
/// Converts external contact wrench $\mathbf{F}_{ext}$ into smooth velocity commands:
/// $\mathbf{M}_d \Delta\ddot{\mathbf{x}} + \mathbf{D}_d \Delta\dot{\mathbf{x}} = \mathbf{F}_{ext}$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmittanceController {
    /// Virtual mass [kg, kg*m^2].
    pub mass: Cartesian6D,
    /// Virtual damping [N*s/m, N*m*s/rad].
    pub damping: Cartesian6D,
    /// Current velocity delta state.
    velocity_delta: Cartesian6D,
}

impl Default for AdmittanceController {
    fn default() -> Self {
        Self {
            mass: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5],
            damping: [25.0, 25.0, 25.0, 2.5, 2.5, 2.5],
            velocity_delta: [0.0; 6],
        }
    }
}

impl AdmittanceController {
    /// Creates a new admittance controller.
    pub const fn new(mass: Cartesian6D, damping: Cartesian6D) -> Self {
        Self {
            mass,
            damping,
            velocity_delta: [0.0; 6],
        }
    }

    /// Resets internal velocity delta state to zero.
    pub fn reset(&mut self) {
        self.velocity_delta = [0.0; 6];
    }

    /// Updates admittance state with incoming external wrench and returns updated Cartesian velocity adjustments.
    pub fn update(&mut self, f_ext: &Cartesian6D, dt: f64) -> Cartesian6D {
        for i in 0..6 {
            let m = if self.mass[i] < 1e-4 {
                1e-4
            } else {
                self.mass[i]
            };
            // a = (F_ext - D * v) / M
            let accel = (f_ext[i] - self.damping[i] * self.velocity_delta[i]) / m;
            self.velocity_delta[i] += accel * dt;
        }
        self.velocity_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impedance_wrench_calculation() {
        let imp = ImpedanceController::default();
        let target_x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let actual_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let zero_vel = [0.0; 6];
        let zero_acc = [0.0; 6];

        let wrench = imp.compute_cartesian_wrench(
            &target_x, &actual_x, &zero_vel, &zero_vel, &zero_acc, &zero_acc,
        );

        // K_p = 500, error = 0.1 -> Fx = 50 N
        assert!((wrench[0] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_admittance_lead_through() {
        let mut adm = AdmittanceController::default();
        let f_push = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 10 N push along X
        let dt = 0.001;

        // Push for 1 second (1000 steps)
        for _ in 0..1000 {
            adm.update(&f_push, dt);
        }

        // Terminal velocity should reach F / D = 10.0 / 25.0 = 0.4 m/s
        let v = adm.update(&f_push, dt);
        assert!((v[0] - 0.4).abs() < 1e-2);
    }
}
