//! Geometric Jacobian, manipulability measures, and DLS singularity attenuation.

use crate::link::Link;
use crate::spatial::{Mat3, Vec3};

/// 6xN Geometric Jacobian matrix represented in base frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricJacobian<const N: usize> {
    /// Linear velocity components [vx, vy, vz] for each joint column.
    pub j_v: [[f64; N]; 3],
    /// Angular velocity components [wx, wy, wz] for each joint column.
    pub j_w: [[f64; N]; 3],
}

impl<const N: usize> Default for GeometricJacobian<N> {
    fn default() -> Self {
        Self {
            j_v: [[0.0; N]; 3],
            j_w: [[0.0; N]; 3],
        }
    }
}

impl<const N: usize> GeometricJacobian<N> {
    /// Computes the Geometric Jacobian for an N-link serial manipulator.
    ///
    /// Expresses end-effector linear and angular velocity in base frame coordinates:
    /// `v_e = J * q_dot`
    pub fn compute(links: &[Link; N], q: &[f64; N]) -> Self {
        let mut t_rot = [Mat3::identity(); N];
        let mut p_orig = [Vec3::zero(); N];
        let mut z_axis = [Vec3::new(0.0, 0.0, 1.0); N];

        // Cumulative transformation from base frame 0 to each frame i
        let mut cum_rot = Mat3::identity();
        let mut cum_pos = Vec3::zero();

        for i in 0..N {
            let theta = q[i] + links[i].theta_offset;
            let rot_i = Mat3::from_dh(theta, links[i].alpha);
            let p_i = links[i].link_offset_vector();

            // Record joint axis z_{i-1} and position p_{i-1} in base frame before multiplying this link
            z_axis[i] = cum_rot.mul_vec(&Vec3::new(0.0, 0.0, 1.0));
            p_orig[i] = cum_pos;

            // Update cumulative transformation
            cum_pos = cum_pos.add(&cum_rot.mul_vec(&p_i));
            cum_rot = cum_rot.mul_mat(&rot_i);
            t_rot[i] = cum_rot;
        }

        let p_end_effector = cum_pos;
        let mut j_v = [[0.0; N]; 3];
        let mut j_w = [[0.0; N]; 3];

        for i in 0..N {
            // j_v[:, i] = z_{i-1} x (p_e - p_{i-1})
            let delta_p = p_end_effector.sub(&p_orig[i]);
            let cross = z_axis[i].cross(&delta_p);

            j_v[0][i] = cross.x();
            j_v[1][i] = cross.y();
            j_v[2][i] = cross.z();

            j_w[0][i] = z_axis[i].x();
            j_w[1][i] = z_axis[i].y();
            j_w[2][i] = z_axis[i].z();
        }

        Self { j_v, j_w }
    }

    /// Multiplies the Jacobian by joint velocities `qd` to yield end-effector Cartesian twist `[v_e, w_e]`.
    pub fn forward_velocity(&self, qd: &[f64; N]) -> [f64; 6] {
        let mut twist = [0.0; 6];
        for i in 0..N {
            twist[0] += self.j_v[0][i] * qd[i];
            twist[1] += self.j_v[1][i] * qd[i];
            twist[2] += self.j_v[2][i] * qd[i];

            twist[3] += self.j_w[0][i] * qd[i];
            twist[4] += self.j_w[1][i] * qd[i];
            twist[5] += self.j_w[2][i] * qd[i];
        }
        twist
    }

    /// Multiplies Jacobian transpose $J^T$ by Cartesian wrench $\mathbf{F} = [f_x, f_y, f_z, n_x, n_y, n_z]^T$
    /// to yield required joint torques $\tau = J^T \mathbf{F}$.
    pub fn transpose_wrench(&self, wrench: &[f64; 6]) -> [f64; N] {
        let mut torques = [0.0; N];
        for i in 0..N {
            torques[i] = self.j_v[0][i] * wrench[0]
                + self.j_v[1][i] * wrench[1]
                + self.j_v[2][i] * wrench[2]
                + self.j_w[0][i] * wrench[3]
                + self.j_w[1][i] * wrench[4]
                + self.j_w[2][i] * wrench[5];
        }
        torques
    }

    /// Computes the Yoshikawa positional manipulability index:
    /// $w_v = \sqrt{\det(J_v J_v^T)}$
    pub fn positional_manipulability(&self) -> f64 {
        // Compute 3x3 matrix M = J_v * J_v^T
        let mut m = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += self.j_v[r][k] * self.j_v[c][k];
                }
                m[r][c] = sum;
            }
        }

        let mat = Mat3 { data: m };
        let det = mat.det();
        if det <= 0.0 {
            0.0
        } else {
            #[cfg(feature = "std")]
            return det.sqrt();
            #[cfg(not(feature = "std"))]
            return libm::sqrt(det);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2link_planar_jacobian() {
        // 2-DOF planar arm with l1 = 1.0, l2 = 1.0
        let mut links = [Link::default(); 2];
        links[0].a = 1.0;
        links[1].a = 1.0;

        let q = [0.0, 0.0]; // Extended along X axis
        let j = GeometricJacobian::<2>::compute(&links, &q);

        // At q = [0, 0], end-effector is at [2, 0, 0]
        // Joint 0 is at [0, 0, 0], z_0 = [0, 0, 1] -> z_0 x [2, 0, 0] = [0, 2, 0]
        // Joint 1 is at [1, 0, 0], z_1 = [0, 0, 1] -> z_1 x [1, 0, 0] = [0, 1, 0]
        assert!((j.j_v[1][0] - 2.0).abs() < 1e-4);
        assert!((j.j_v[1][1] - 1.0).abs() < 1e-4);

        // At full extension (singularity in X motion), positional manipulability in 3D has rank 2
        let w = j.positional_manipulability();
        assert_eq!(w, 0.0); // Z is uncontrolled in 2D planar arm
    }
}
