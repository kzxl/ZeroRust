//! Deterministic 7-phase Jerk-limited S-Curve trajectory generator.
//!
//! Generates $C^2$-continuous position, velocity, and acceleration profiles with bounded jerk
//! for smooth vibration-free robotic and CNC motion.

use zero_core::error::{ZeroError, ZeroResult};

/// Physical kinematics constraints for trajectory generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SCurveConstraints {
    /// Maximum allowable velocity (units/s).
    pub max_velocity: f64,
    /// Maximum allowable acceleration/deceleration (units/s^2).
    pub max_acceleration: f64,
    /// Maximum allowable jerk (units/s^3).
    pub max_jerk: f64,
}

/// Instantaneous trajectory state at sampling instant $t$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TrajectoryPoint {
    /// Commanded position $q(t)$.
    pub position: f64,
    /// Commanded velocity $\dot{q}(t)$.
    pub velocity: f64,
    /// Commanded acceleration $\ddot{q}(t)$.
    pub acceleration: f64,
    /// Commanded jerk $\dddot{q}(t)$.
    pub jerk: f64,
}

/// A planned 7-phase S-curve trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SCurveTrajectory {
    q0: f64,
    direction: f64,
    tj: f64, // Duration of jerk phases (1, 3, 5, 7)
    ta: f64, // Duration of constant acceleration phase (2)
    tv: f64, // Duration of constant velocity phase (4)
    td: f64, // Duration of constant deceleration phase (6)
    v_peak: f64,
    a_peak: f64,
    j_max: f64,
    total_time: f64,
}

impl SCurveTrajectory {
    /// Returns the total execution time of the trajectory in seconds.
    #[inline]
    pub const fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Evaluates the trajectory state at time `t` (seconds from motion start).
    pub fn sample(&self, t: f64) -> TrajectoryPoint {
        if t <= 0.0 {
            return TrajectoryPoint {
                position: self.q0,
                velocity: 0.0,
                acceleration: 0.0,
                jerk: 0.0,
            };
        }
        if t >= self.total_time {
            let total_dist = self.sample_displacement(self.total_time);
            return TrajectoryPoint {
                position: self.q0 + self.direction * total_dist,
                velocity: 0.0,
                acceleration: 0.0,
                jerk: 0.0,
            };
        }

        let (dist, vel, acc, jerk) = self.eval_1d(t);
        TrajectoryPoint {
            position: self.q0 + self.direction * dist,
            velocity: self.direction * vel,
            acceleration: self.direction * acc,
            jerk: self.direction * jerk,
        }
    }

    fn sample_displacement(&self, t: f64) -> f64 {
        let (dist, _, _, _) = self.eval_1d(t);
        dist
    }

    fn eval_1d(&self, t: f64) -> (f64, f64, f64, f64) {
        let t1 = self.tj;
        let t2 = t1 + self.ta;
        let t3 = t2 + self.tj;
        let t4 = t3 + self.tv;
        let t5 = t4 + self.tj;
        let t6 = t5 + self.td;
        let _t7 = self.total_time;

        let j = self.j_max;
        let a = self.a_peak;
        let v = self.v_peak;

        // End of Phase 1
        let v1 = 0.5 * j * t1 * t1;
        let s1 = (1.0 / 6.0) * j * t1 * t1 * t1;

        // End of Phase 2
        let dt2 = self.ta;
        let v2 = v1 + a * dt2;
        let s2 = s1 + v1 * dt2 + 0.5 * a * dt2 * dt2;

        // End of Phase 3
        let dt3 = self.tj;
        let v3 = v2 + a * dt3 - 0.5 * j * dt3 * dt3;
        let s3 = s2 + v2 * dt3 + 0.5 * a * dt3 * dt3 - (1.0 / 6.0) * j * dt3 * dt3 * dt3;

        // End of Phase 4
        let dt4 = self.tv;
        let s4 = s3 + v3 * dt4;

        if t <= t1 {
            // Phase 1: Jerk = +j
            let pos = (1.0 / 6.0) * j * t * t * t;
            let vel = 0.5 * j * t * t;
            let acc = j * t;
            (pos, vel, acc, j)
        } else if t <= t2 {
            // Phase 2: Jerk = 0, Acc = a
            let dt = t - t1;
            let pos = s1 + v1 * dt + 0.5 * a * dt * dt;
            let vel = v1 + a * dt;
            let acc = a;
            (pos, vel, acc, 0.0)
        } else if t <= t3 {
            // Phase 3: Jerk = -j
            let dt = t - t2;
            let pos = s2 + v2 * dt + 0.5 * a * dt * dt - (1.0 / 6.0) * j * dt * dt * dt;
            let vel = v2 + a * dt - 0.5 * j * dt * dt;
            let acc = a - j * dt;
            (pos, vel, acc, -j)
        } else if t <= t4 {
            // Phase 4: Constant velocity
            let dt = t - t3;
            let pos = s3 + v * dt;
            let vel = v;
            (pos, vel, 0.0, 0.0)
        } else if t <= t5 {
            // Phase 5: Jerk = -j (Deceleration build up)
            let dt = t - t4;
            let pos = s4 + v * dt - (1.0 / 6.0) * j * dt * dt * dt;
            let vel = v - 0.5 * j * dt * dt;
            let acc = -j * dt;
            (pos, vel, acc, -j)
        } else if t <= t6 {
            // Phase 6: Constant deceleration = -a
            let dt = t - t5;
            let dt5 = self.tj;
            let v5 = v - 0.5 * j * dt5 * dt5;
            let s5 = s4 + v * dt5 - (1.0 / 6.0) * j * dt5 * dt5 * dt5;

            let pos = s5 + v5 * dt - 0.5 * a * dt * dt;
            let vel = v5 - a * dt;
            let acc = -a;
            (pos, vel, acc, 0.0)
        } else {
            // Phase 7: Jerk = +j (Deceleration ramp down to 0)
            let dt = t - t6;
            let dt5 = self.tj;
            let v5 = v - 0.5 * j * dt5 * dt5;
            let s5 = s4 + v * dt5 - (1.0 / 6.0) * j * dt5 * dt5 * dt5;

            let dt6 = self.td;
            let v6 = v5 - a * dt6;
            let s6 = s5 + v5 * dt6 - 0.5 * a * dt6 * dt6;

            let pos = s6 + v6 * dt - 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt * dt * dt;
            let vel = v6 - a * dt + 0.5 * j * dt * dt;
            let acc = -a + j * dt;
            (pos, vel, acc, j)
        }
    }
}

/// S-Curve Trajectory Planner.
pub struct SCurvePlanner {
    constraints: SCurveConstraints,
}

impl SCurvePlanner {
    /// Creates a new planner with given motion constraints.
    pub const fn new(constraints: SCurveConstraints) -> Self {
        Self { constraints }
    }

    /// Plans a point-to-point motion from `q0` to `q1`.
    pub fn plan(&self, q0: f64, q1: f64) -> ZeroResult<SCurveTrajectory> {
        let delta = q1 - q0;
        let l = delta.abs();
        if l < 1e-9 {
            return Ok(SCurveTrajectory {
                q0,
                direction: 1.0,
                tj: 0.0,
                ta: 0.0,
                tv: 0.0,
                td: 0.0,
                v_peak: 0.0,
                a_peak: 0.0,
                j_max: self.constraints.max_jerk,
                total_time: 0.0,
            });
        }

        let direction = if delta >= 0.0 { 1.0 } else { -1.0 };
        let j = self.constraints.max_jerk;
        let mut a_max = self.constraints.max_acceleration;
        let v_max = self.constraints.max_velocity;

        if j <= 0.0 || a_max <= 0.0 || v_max <= 0.0 {
            return Err(ZeroError::InvalidArgument);
        }

        // Check if a_max is reachable
        let mut tj = a_max / j;
        if v_max < a_max * tj {
            #[cfg(feature = "std")]
            {
                tj = (v_max / j).sqrt();
            }
            #[cfg(not(feature = "std"))]
            {
                tj = libm::sqrt(v_max / j);
            }
            a_max = tj * j;
        }

        let ta = (v_max - a_max * tj) / a_max;

        // Distance during acceleration phase
        let sa = a_max * tj * (tj + ta);

        let tv = if l > 2.0 * sa {
            (l - 2.0 * sa) / v_max
        } else {
            // Need to reduce v_max because distance is too short
            0.0
        };

        let (final_tj, final_ta, final_tv, final_v, final_a) = if tv == 0.0 {
            // Short move: solve for reachable velocity
            let mut v = v_max;
            let mut tj_s = tj;
            let mut ta_s = ta;

            for _ in 0..10 {
                let s_acc = a_max * tj_s * (tj_s + ta_s);
                if s_acc > l * 0.5 {
                    v *= 0.8;
                    if v < a_max * tj_s {
                        #[cfg(feature = "std")]
                        {
                            tj_s = (v / j).sqrt();
                        }
                        #[cfg(not(feature = "std"))]
                        {
                            tj_s = libm::sqrt(v / j);
                        }
                        a_max = tj_s * j;
                    }
                    ta_s = ((v - a_max * tj_s) / a_max).max(0.0);
                } else {
                    break;
                }
            }
            (tj_s, ta_s, 0.0, v, a_max)
        } else {
            (tj, ta, tv, v_max, a_max)
        };

        let td = final_ta;
        let total_time = 4.0 * final_tj + 2.0 * final_ta + final_tv;

        Ok(SCurveTrajectory {
            q0,
            direction,
            tj: final_tj,
            ta: final_ta,
            tv: final_tv,
            td,
            v_peak: final_v,
            a_peak: final_a,
            j_max: j,
            total_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scurve_trajectory_planning() {
        let constraints = SCurveConstraints {
            max_velocity: 100.0,
            max_acceleration: 500.0,
            max_jerk: 2000.0,
        };

        let planner = SCurvePlanner::new(constraints);
        let traj = planner.plan(0.0, 50.0).unwrap();

        assert!(traj.total_time() > 0.0);

        let p_start = traj.sample(0.0);
        assert_eq!(p_start.position, 0.0);
        assert_eq!(p_start.velocity, 0.0);

        let p_end = traj.sample(traj.total_time());
        assert!((p_end.position - 50.0).abs() < 1.0);
    }
}
