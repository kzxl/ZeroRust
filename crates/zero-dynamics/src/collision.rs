//! Sensorless collision detection via generalized momentum disturbance observer (Haddadin / De Luca).

use crate::link::Link;
use crate::rnea::RneaSolver;
use crate::spatial::Vec3;

/// Status and diagnostic details of collision detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionStatus {
    /// True if an external collision or unexpected contact is active.
    pub collision_detected: bool,
    /// Joint index that triggered the threshold breach (if any).
    pub fault_joint: Option<usize>,
    /// Peak estimated external collision torque [N*m].
    pub peak_torque: f64,
}

/// Generalized momentum observer for sensorless contact and collision monitoring.
#[derive(Debug, Clone)]
pub struct MomentumObserver<const N: usize> {
    /// Observer gain diagonal coefficients $K_I$ (typically 20.0 - 100.0 rad/s).
    pub observer_gain: [f64; N],
    /// Torque thresholds for triggering safety stop [N*m].
    pub thresholds: [f64; N],
    /// Directional base acceleration vector for gravity compensation.
    pub gravity: Vec3,
    /// Accumulated discrete integral state.
    integral: [f64; N],
    /// Initial momentum at initialization: $p(0) = M(q_0)\dot{q}_0$.
    initial_p: [f64; N],
    /// Filtered residual disturbance torque estimate: $r \approx \tau_{ext}$.
    residual: [f64; N],
    /// True if initialized.
    initialized: bool,
}

impl<const N: usize> MomentumObserver<N> {
    /// Creates a new momentum observer with given gain, joint safety thresholds, and gravity vector.
    pub const fn new(gain: [f64; N], thresholds: [f64; N], gravity: Vec3) -> Self {
        Self {
            observer_gain: gain,
            thresholds,
            gravity,
            integral: [0.0; N],
            initial_p: [0.0; N],
            residual: [0.0; N],
            initialized: false,
        }
    }

    /// Resets the observer state to current robot pose.
    pub fn reset(
        &mut self,
        solver: &mut RneaSolver<N>,
        links: &[Link; N],
        q: &[f64; N],
        qd: &[f64; N],
    ) {
        let mass_mat = solver.compute_mass_matrix(links, q);
        for i in 0..N {
            let mut p_i = 0.0;
            for j in 0..N {
                p_i += mass_mat[i][j] * qd[j];
            }
            self.initial_p[i] = p_i;
            self.integral[i] = 0.0;
            self.residual[i] = 0.0;
        }
        self.initialized = true;
    }

    /// Evaluates one discrete time step of the momentum observer.
    ///
    /// # Arguments
    /// - `solver`: RNEA dynamics solver.
    /// - `links`: Link parameter array.
    /// - `q`: Joint positions (rad).
    /// - `qd`: Joint velocities (rad/s).
    /// - `tau_feedback`: Measured motor torque feedback (from drives).
    /// - `dt`: Cycle time in seconds (e.g. 0.001 for 1 kHz).
    pub fn update(
        &mut self,
        solver: &mut RneaSolver<N>,
        links: &[Link; N],
        q: &[f64; N],
        qd: &[f64; N],
        tau_feedback: &[f64; N],
        dt: f64,
    ) -> CollisionStatus {
        if !self.initialized {
            self.reset(solver, links, q, qd);
        }

        // 1. Current generalized momentum: p = M(q) * qd
        let mass_mat = solver.compute_mass_matrix(links, q);
        let mut p = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                p[i] += mass_mat[i][j] * qd[j];
            }
        }

        // 2. Coriolis and gravity compensation terms
        let g = solver.compute_gravity(links, q, self.gravity);
        let c_qd = solver.compute_coriolis(links, q, qd);

        let mut peak_torque: f64 = 0.0;
        let mut fault_joint = None;
        let mut collision_detected = false;

        // 3. Update discrete integral and residual for each joint
        for i in 0..N {
            // beta = tau_feedback - (c_qd + g[i]) + residual[i]
            // Note: in RNEA, g includes positive torque to support arm against gravity.
            let model_feedforward = c_qd[i] + g[i];
            let dot_term = tau_feedback[i] - model_feedforward + self.residual[i];

            self.integral[i] += dot_term * dt;

            // r(t) = K_I * [ p(t) - integral - p(0) ]
            self.residual[i] =
                self.observer_gain[i] * (p[i] - self.integral[i] - self.initial_p[i]);

            #[cfg(feature = "std")]
            let abs_res = self.residual[i].abs();
            #[cfg(not(feature = "std"))]
            let abs_res = libm::fabs(self.residual[i]);

            if abs_res > peak_torque {
                peak_torque = abs_res;
            }

            if abs_res > self.thresholds[i] {
                collision_detected = true;
                if fault_joint.is_none() {
                    fault_joint = Some(i);
                }
            }
        }

        CollisionStatus {
            collision_detected,
            fault_joint,
            peak_torque,
        }
    }

    /// Current filtered disturbance torque estimate [N*m] across all joints.
    pub const fn estimated_external_torque(&self) -> &[f64; N] {
        &self.residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::InertiaTensor;
    use crate::spatial::Vec3;

    #[test]
    fn test_collision_detection_trigger() {
        let mut links = [Link::default(); 1];
        links[0].a = 0.5;
        links[0].mass = 1.0;
        links[0].com = Vec3::new(0.25, 0.0, 0.0);
        links[0].inertia = InertiaTensor::principal(0.01, 0.05, 0.05);

        let mut solver = RneaSolver::<1>::new();
        let gravity = Vec3::new(0.0, 9.80665, 0.0);
        let mut observer = MomentumObserver::new([30.0], [5.0], gravity); // 5 Nm collision threshold

        let q = [0.0];
        let qd = [0.0];
        observer.reset(&mut solver, &links, &q, &qd);

        // Stationary arm holding against gravity: model torque = 1.0 * 9.80665 * 0.25 = 2.4516 Nm
        let tau_nominal = [2.4516625];
        let dt = 0.001;

        // Normal operation for 100 steps
        for _ in 0..100 {
            let status = observer.update(&mut solver, &links, &q, &qd, &tau_nominal, dt);
            assert!(!status.collision_detected);
        }

        // Sudden external collision force: motor current spikes by +10.0 Nm
        let tau_collision = [tau_nominal[0] + 10.0];
        let mut tripped = false;
        for _ in 0..50 {
            let status = observer.update(&mut solver, &links, &q, &qd, &tau_collision, dt);
            if status.collision_detected {
                tripped = true;
                assert_eq!(status.fault_joint, Some(0));
                break;
            }
        }
        assert!(tripped);
    }
}
