//! Recursive Newton-Euler Algorithm (RNEA) $O(N)$ inverse dynamics engine.

use crate::link::Link;
use crate::spatial::{Mat3, Vec3};

/// Workspace buffers for RNEA forward and backward passes without heap allocation.
#[derive(Debug, Clone)]
pub struct RneaSolver<const N: usize> {
    rot: [Mat3; N],
    p: [Vec3; N],
    omega: [Vec3; N],
    alpha: [Vec3; N],
    a: [Vec3; N],
    a_com: [Vec3; N],
    f_inertial: [Vec3; N],
    n_inertial: [Vec3; N],
    f_internal: [Vec3; N],
    n_internal: [Vec3; N],
}

impl<const N: usize> Default for RneaSolver<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> RneaSolver<N> {
    /// Creates a new RNEA solver for an N-link serial manipulator.
    pub const fn new() -> Self {
        Self {
            rot: [Mat3::identity(); N],
            p: [Vec3::zero(); N],
            omega: [Vec3::zero(); N],
            alpha: [Vec3::zero(); N],
            a: [Vec3::zero(); N],
            a_com: [Vec3::zero(); N],
            f_inertial: [Vec3::zero(); N],
            n_inertial: [Vec3::zero(); N],
            f_internal: [Vec3::zero(); N],
            n_internal: [Vec3::zero(); N],
        }
    }

    /// Solves the full inverse dynamics using Recursive Newton-Euler Algorithm (RNEA).
    ///
    /// Returns joint torques $\tau_i = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) + \tau_{friction}$.
    ///
    /// # Arguments
    /// - `links`: Array of link inertial & geometric parameters.
    /// - `q`: Joint positions (radians).
    /// - `qd`: Joint velocities (rad/s).
    /// - `qdd`: Joint accelerations (rad/s^2).
    /// - `gravity`: Spatial base acceleration vector (e.g. `[0.0, 0.0, 9.80665]` for upright Z gravity).
    /// - `f_ext`: Optional external wrench applied at end-effector `[fx, fy, fz, nx, ny, nz]`.
    pub fn solve(
        &mut self,
        links: &[Link; N],
        q: &[f64; N],
        qd: &[f64; N],
        qdd: &[f64; N],
        gravity: Vec3,
        f_ext: Option<[f64; 6]>,
    ) -> [f64; N] {
        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        // Precompute transformation matrices R_i^{i-1} and translation p_i
        for i in 0..N {
            let theta = q[i] + links[i].theta_offset;
            self.rot[i] = Mat3::from_dh(theta, links[i].alpha);
            self.p[i] = links[i].link_offset_vector();
        }

        // ==========================================
        // Forward Pass: Kinematics (Base -> Tip)
        // ==========================================
        for i in 0..N {
            let r_t = self.rot[i].transpose(); // R_{i-1}^i (transforms vector from frame i-1 to frame i)

            let prev_omega = if i == 0 {
                Vec3::zero()
            } else {
                self.omega[i - 1]
            };

            let prev_alpha = if i == 0 {
                Vec3::zero()
            } else {
                self.alpha[i - 1]
            };

            // omega_i = R^T * omega_{i-1} + qd_i * z_0
            let r_prev_omega = r_t.mul_vec(&prev_omega);
            self.omega[i] = r_prev_omega.add(&z_axis.scale(qd[i]));

            // alpha_i = R^T * alpha_{i-1} + qdd_i * z_0 + (R^T * omega_{i-1}) x (qd_i * z_0)
            let r_prev_alpha = r_t.mul_vec(&prev_alpha);
            let cross_term = r_prev_omega.cross(&z_axis.scale(qd[i]));
            self.alpha[i] = r_prev_alpha.add(&z_axis.scale(qdd[i])).add(&cross_term);

            // Linear acceleration of joint i origin
            self.a[i] = if i == 0 {
                r_t.mul_vec(&gravity)
            } else {
                let alpha_prev_x_p = prev_alpha.cross(&self.p[i]);
                let omega_prev_x_p = prev_omega.cross(&self.p[i]);
                let omega_x_omega_x_p = prev_omega.cross(&omega_prev_x_p);
                let a_prev_transferred = self.a[i - 1].add(&alpha_prev_x_p).add(&omega_x_omega_x_p);
                r_t.mul_vec(&a_prev_transferred)
            };

            // a_ci = a_i + alpha_i x r_ci + omega_i x (omega_i x r_ci)
            let r_ci = links[i].com;
            let alpha_x_com = self.alpha[i].cross(&r_ci);
            let omega_x_com = self.omega[i].cross(&r_ci);
            let omega_x_omega_x_com = self.omega[i].cross(&omega_x_com);
            self.a_com[i] = self.a[i].add(&alpha_x_com).add(&omega_x_omega_x_com);

            // Inertial force & torque
            // F_i = m_i * a_ci
            self.f_inertial[i] = self.a_com[i].scale(links[i].mass);

            // N_i = I_i * alpha_i + omega_i x (I_i * omega_i)
            let i_alpha = links[i].inertia.mat.mul_vec(&self.alpha[i]);
            let i_omega = links[i].inertia.mat.mul_vec(&self.omega[i]);
            let omega_x_i_omega = self.omega[i].cross(&i_omega);
            self.n_inertial[i] = i_alpha.add(&omega_x_i_omega);
        }

        // ==========================================
        // Backward Pass: Kinetics (Tip -> Base)
        // ==========================================
        let mut torques = [0.0; N];

        for i in (0..N).rev() {
            let (f_next, n_next) = if i == N - 1 {
                if let Some(ext) = f_ext {
                    (
                        Vec3::new(ext[0], ext[1], ext[2]),
                        Vec3::new(ext[3], ext[4], ext[5]),
                    )
                } else {
                    (Vec3::zero(), Vec3::zero())
                }
            } else {
                let r_next = self.rot[i + 1];
                let f_rotated = r_next.mul_vec(&self.f_internal[i + 1]);
                let n_rotated = r_next.mul_vec(&self.n_internal[i + 1]);
                (f_rotated, n_rotated)
            };

            // f_i = f_next + F_i
            self.f_internal[i] = f_next.add(&self.f_inertial[i]);

            // n_i = N_i + n_next + r_ci x F_i + p_{i+1} x f_next
            let r_ci_x_f = links[i].com.cross(&self.f_inertial[i]);
            let p_next_x_f = if i < N - 1 {
                self.p[i + 1].cross(&f_next)
            } else {
                Vec3::zero()
            };

            self.n_internal[i] = self.n_inertial[i]
                .add(&n_next)
                .add(&r_ci_x_f)
                .add(&p_next_x_f);

            // Joint torque: tau_i = n_i . z_0 + friction
            let tau_dynamic = self.n_internal[i].dot(&z_axis);
            let tau_friction = links[i].friction_torque(qd[i]);
            torques[i] = tau_dynamic + tau_friction;
        }

        torques
    }

    /// Computes only gravity torques: g(q).
    pub fn compute_gravity(&mut self, links: &[Link; N], q: &[f64; N], gravity: Vec3) -> [f64; N] {
        let zero_vel = [0.0; N];
        let zero_acc = [0.0; N];
        self.solve(links, q, &zero_vel, &zero_acc, gravity, None)
    }

    /// Computes only Coriolis and centrifugal torques: C(q, qd) * qd.
    pub fn compute_coriolis(&mut self, links: &[Link; N], q: &[f64; N], qd: &[f64; N]) -> [f64; N] {
        let zero_acc = [0.0; N];
        let zero_g = Vec3::zero();
        self.solve(links, q, qd, &zero_acc, zero_g, None)
    }

    /// Computes the complete symmetric positive-definite Mass Matrix M(q).
    pub fn compute_mass_matrix(&mut self, links: &[Link; N], q: &[f64; N]) -> [[f64; N]; N] {
        let zero_vel = [0.0; N];
        let zero_g = Vec3::zero();
        let mut mass_matrix = [[0.0; N]; N];

        for col in 0..N {
            let mut unit_acc = [0.0; N];
            unit_acc[col] = 1.0;
            let col_torques = self.solve(links, q, &zero_vel, &unit_acc, zero_g, None);
            for row in 0..N {
                mass_matrix[row][col] = col_torques[row];
            }
        }

        mass_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::InertiaTensor;

    #[test]
    fn test_single_pendulum_gravity() {
        // 1-DOF pendulum: link length = 1.0 m, mass = 2.0 kg, CoM at [0.5, 0, 0]
        let mut links = [Link::default(); 1];
        links[0].a = 1.0;
        links[0].mass = 2.0;
        links[0].com = Vec3::new(0.5, 0.0, 0.0);
        links[0].inertia = InertiaTensor::principal(0.1, 0.5, 0.5);

        let mut solver = RneaSolver::<1>::new();

        // Horizontal position (q = 0 rad): with gravity along Y (perpendicular to Z axis),
        // gravity torque = m * g * l_c = 2.0 * 9.81 * 0.5 = 9.81 Nm
        let q_horizontal = [0.0];
        let tau_g = solver.compute_gravity(&links, &q_horizontal, Vec3::new(0.0, 9.81, 0.0));
        assert!((tau_g[0] - 9.81).abs() < 1e-3);

        // Mass matrix should equal I_zz + m * l_c^2 = 0.5 + 2.0 * 0.25 = 1.0 kg*m^2
        let m = solver.compute_mass_matrix(&links, &q_horizontal);
        assert!((m[0][0] - 1.0).abs() < 1e-3);
    }
}
