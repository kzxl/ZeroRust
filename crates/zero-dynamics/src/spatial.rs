//! Spatial math primitives: 3D vectors, 3x3 matrices, and rotation transformations.

/// 3D Vector with zero-allocation math operations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// Vector components [x, y, z].
    pub data: [f64; 3],
}

impl Vec3 {
    /// Creates a new 3D vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { data: [x, y, z] }
    }

    /// Zero vector [0, 0, 0].
    pub const fn zero() -> Self {
        Self {
            data: [0.0, 0.0, 0.0],
        }
    }

    /// X coordinate.
    pub const fn x(&self) -> f64 {
        self.data[0]
    }

    /// Y coordinate.
    pub const fn y(&self) -> f64 {
        self.data[1]
    }

    /// Z coordinate.
    pub const fn z(&self) -> f64 {
        self.data[2]
    }

    /// Vector addition.
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.data[0] + other.data[0],
            self.data[1] + other.data[1],
            self.data[2] + other.data[2],
        )
    }

    /// Vector subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(
            self.data[0] - other.data[0],
            self.data[1] - other.data[1],
            self.data[2] - other.data[2],
        )
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.data[0] * s, self.data[1] * s, self.data[2] * s)
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f64 {
        self.data[0] * other.data[0] + self.data[1] * other.data[1] + self.data[2] * other.data[2]
    }

    /// Cross product: self x other.
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        )
    }

    /// Squared Euclidean norm.
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        #[cfg(feature = "std")]
        return self.norm_squared().sqrt();
        #[cfg(not(feature = "std"))]
        return libm::sqrt(self.norm_squared());
    }
}

/// 3x3 Matrix for rotation tensors and rotational inertia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    /// Row-major 3x3 elements: [row 0, row 1, row 2].
    pub data: [[f64; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat3 {
    /// 3x3 Identity matrix.
    pub const fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// 3x3 Zero matrix.
    pub const fn zero() -> Self {
        Self {
            data: [[0.0; 3]; 3],
        }
    }

    /// Matrix-vector multiplication: M * v.
    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            self.data[0][0] * v.x() + self.data[0][1] * v.y() + self.data[0][2] * v.z(),
            self.data[1][0] * v.x() + self.data[1][1] * v.y() + self.data[1][2] * v.z(),
            self.data[2][0] * v.x() + self.data[2][1] * v.y() + self.data[2][2] * v.z(),
        )
    }

    /// Matrix transpose: M^T.
    pub fn transpose(&self) -> Self {
        Self {
            data: [
                [self.data[0][0], self.data[1][0], self.data[2][0]],
                [self.data[0][1], self.data[1][1], self.data[2][1]],
                [self.data[0][2], self.data[1][2], self.data[2][2]],
            ],
        }
    }

    /// Matrix multiplication: self * other.
    pub fn mul_mat(&self, other: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][0] * other.data[0][j]
                    + self.data[i][1] * other.data[1][j]
                    + self.data[i][2] * other.data[2][j];
            }
        }
        Self { data: out }
    }

    /// Computes the 3x3 Rotation Matrix from standard DH parameters (theta, alpha).
    ///
    /// R_i^{i-1} transforms from frame i to frame i-1:
    /// [ cos(theta)   -sin(theta)*cos(alpha)    sin(theta)*sin(alpha) ]
    /// [ sin(theta)    cos(theta)*cos(alpha)   -cos(theta)*sin(alpha) ]
    /// [     0              sin(alpha)               cos(alpha)       ]
    pub fn from_dh(theta: f64, alpha: f64) -> Self {
        #[cfg(feature = "std")]
        let (st, ct) = (theta.sin(), theta.cos());
        #[cfg(feature = "std")]
        let (sa, ca) = (alpha.sin(), alpha.cos());

        #[cfg(not(feature = "std"))]
        let (st, ct) = (libm::sin(theta), libm::cos(theta));
        #[cfg(not(feature = "std"))]
        let (sa, ca) = (libm::sin(alpha), libm::cos(alpha));

        Self {
            data: [
                [ct, -st * ca, st * sa],
                [st, ct * ca, -ct * sa],
                [0.0, sa, ca],
            ],
        }
    }

    /// Computes determinant of 3x3 matrix.
    pub fn det(&self) -> f64 {
        let m = &self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Computes inverse of 3x3 matrix. Returns None if determinant is near-zero.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.det();
        #[cfg(feature = "std")]
        let is_singular = det.abs() < 1e-12;
        #[cfg(not(feature = "std"))]
        let is_singular = libm::fabs(det) < 1e-12;

        if is_singular {
            return None;
        }

        let inv_det = 1.0 / det;
        let m = &self.data;
        let out = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];

        Some(Self { data: out })
    }
}
