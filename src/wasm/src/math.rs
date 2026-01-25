//! Math utilities for Gaussian Splatting
//! Matrix operations, quaternions, and coordinate transforms

use wasm_bindgen::prelude::*;

/// Matrix 4x4 operations
#[wasm_bindgen]
pub struct Mat4 {
    data: [f32; 16],
}

#[wasm_bindgen]
impl Mat4 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::identity()
    }

    #[wasm_bindgen]
    pub fn identity() -> Self {
        Self {
            data: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[wasm_bindgen]
    pub fn from_array(data: &[f32]) -> Self {
        let mut m = Self::identity();
        m.data.copy_from_slice(&data[..16]);
        m
    }

    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    #[wasm_bindgen]
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (near - far);

        Self {
            data: [
                f / aspect,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                (far + near) * nf,
                -1.0,
                0.0,
                0.0,
                2.0 * far * near * nf,
                0.0,
            ],
        }
    }

    #[wasm_bindgen]
    pub fn look_at(
        eye_x: f32,
        eye_y: f32,
        eye_z: f32,
        center_x: f32,
        center_y: f32,
        center_z: f32,
        up_x: f32,
        up_y: f32,
        up_z: f32,
    ) -> Self {
        let fx = center_x - eye_x;
        let fy = center_y - eye_y;
        let fz = center_z - eye_z;

        let f_len = (fx * fx + fy * fy + fz * fz).sqrt();
        let fx = fx / f_len;
        let fy = fy / f_len;
        let fz = fz / f_len;

        // s = f x up
        let sx = fy * up_z - fz * up_y;
        let sy = fz * up_x - fx * up_z;
        let sz = fx * up_y - fy * up_x;

        let s_len = (sx * sx + sy * sy + sz * sz).sqrt();
        let sx = sx / s_len;
        let sy = sy / s_len;
        let sz = sz / s_len;

        // u = s x f
        let ux = sy * fz - sz * fy;
        let uy = sz * fx - sx * fz;
        let uz = sx * fy - sy * fx;

        Self {
            data: [
                sx,
                ux,
                -fx,
                0.0,
                sy,
                uy,
                -fy,
                0.0,
                sz,
                uz,
                -fz,
                0.0,
                -(sx * eye_x + sy * eye_y + sz * eye_z),
                -(ux * eye_x + uy * eye_y + uz * eye_z),
                fx * eye_x + fy * eye_y + fz * eye_z,
                1.0,
            ],
        }
    }

    #[wasm_bindgen]
    pub fn multiply(&self, other: &Mat4) -> Mat4 {
        let mut result = [0.0f32; 16];

        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.data[i + k * 4] * other.data[k + j * 4];
                }
                result[i + j * 4] = sum;
            }
        }

        Mat4 { data: result }
    }

    #[wasm_bindgen]
    pub fn translate(&self, x: f32, y: f32, z: f32) -> Mat4 {
        let t = Mat4 {
            data: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
            ],
        };
        self.multiply(&t)
    }

    #[wasm_bindgen]
    pub fn rotate_y(&self, angle: f32) -> Mat4 {
        let c = angle.cos();
        let s = angle.sin();

        let r = Mat4 {
            data: [
                c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        self.multiply(&r)
    }

    #[wasm_bindgen]
    pub fn invert(&self) -> Option<Mat4> {
        let m = &self.data;
        let mut inv = [0.0f32; 16];

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14]
            + m[13] * m[6] * m[11]
            - m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14]
            - m[12] * m[6] * m[11]
            + m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13]
            + m[12] * m[5] * m[11]
            - m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13]
            - m[12] * m[5] * m[10]
            + m[12] * m[6] * m[9];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if det.abs() < 1e-10 {
            return None;
        }

        // Calculate rest of inverse matrix (abbreviated for space)
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14]
            - m[13] * m[2] * m[11]
            + m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14]
            + m[12] * m[2] * m[11]
            - m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13]
            - m[12] * m[1] * m[11]
            + m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13]
            + m[12] * m[1] * m[10]
            - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14]
            + m[13] * m[2] * m[7]
            - m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14]
            - m[12] * m[2] * m[7]
            + m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13]
            + m[12] * m[1] * m[7]
            - m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13]
            - m[12] * m[1] * m[6]
            + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10]
            - m[9] * m[2] * m[7]
            + m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10]
            + m[8] * m[2] * m[7]
            - m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9]
            - m[8] * m[1] * m[7]
            + m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9]
            + m[8] * m[1] * m[6]
            - m[8] * m[2] * m[5];

        let det_inv = 1.0 / det;
        for i in 0..16 {
            inv[i] *= det_inv;
        }

        Some(Mat4 { data: inv })
    }

    /// Transform a point by this matrix
    #[wasm_bindgen]
    pub fn transform_point(&self, x: f32, y: f32, z: f32) -> Vec<f32> {
        let m = &self.data;
        let w = m[3] * x + m[7] * y + m[11] * z + m[15];
        let w_inv = if w.abs() > 1e-10 { 1.0 / w } else { 1.0 };

        vec![
            (m[0] * x + m[4] * y + m[8] * z + m[12]) * w_inv,
            (m[1] * x + m[5] * y + m[9] * z + m[13]) * w_inv,
            (m[2] * x + m[6] * y + m[10] * z + m[14]) * w_inv,
        ]
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Quaternion operations
#[wasm_bindgen]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[wasm_bindgen]
impl Quaternion {
    #[wasm_bindgen(constructor)]
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    #[wasm_bindgen]
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[wasm_bindgen]
    pub fn from_axis_angle(axis_x: f32, axis_y: f32, axis_z: f32, angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let s = half_angle.sin();

        // Normalize axis
        let len = (axis_x * axis_x + axis_y * axis_y + axis_z * axis_z).sqrt();
        let (ax, ay, az) = if len > 0.0 {
            (axis_x / len, axis_y / len, axis_z / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        Self {
            w: half_angle.cos(),
            x: ax * s,
            y: ay * s,
            z: az * s,
        }
    }

    #[wasm_bindgen]
    pub fn normalize(&mut self) {
        let len = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            let inv_len = 1.0 / len;
            self.w *= inv_len;
            self.x *= inv_len;
            self.y *= inv_len;
            self.z *= inv_len;
        }
    }

    #[wasm_bindgen]
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Convert to 3x3 rotation matrix (column-major, 9 floats)
    #[wasm_bindgen]
    pub fn to_rotation_matrix(&self) -> Vec<f32> {
        let x2 = self.x + self.x;
        let y2 = self.y + self.y;
        let z2 = self.z + self.z;

        let xx2 = self.x * x2;
        let yy2 = self.y * y2;
        let zz2 = self.z * z2;
        let xy2 = self.x * y2;
        let xz2 = self.x * z2;
        let yz2 = self.y * z2;
        let wx2 = self.w * x2;
        let wy2 = self.w * y2;
        let wz2 = self.w * z2;

        vec![
            1.0 - yy2 - zz2,
            xy2 + wz2,
            xz2 - wy2,
            xy2 - wz2,
            1.0 - xx2 - zz2,
            yz2 + wx2,
            xz2 + wy2,
            yz2 - wx2,
            1.0 - xx2 - yy2,
        ]
    }

    /// Spherical linear interpolation
    #[wasm_bindgen]
    pub fn slerp(&self, other: &Quaternion, t: f32) -> Quaternion {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;

        let (ow, ox, oy, oz) = if dot < 0.0 {
            dot = -dot;
            (-other.w, -other.x, -other.y, -other.z)
        } else {
            (other.w, other.x, other.y, other.z)
        };

        let (s0, s1) = if dot > 0.9995 {
            // Linear interpolation for small angles
            (1.0 - t, t)
        } else {
            let theta = dot.acos();
            let sin_theta = theta.sin();
            (((1.0 - t) * theta).sin() / sin_theta, (t * theta).sin() / sin_theta)
        };

        Quaternion {
            w: s0 * self.w + s1 * ow,
            x: s0 * self.x + s1 * ox,
            y: s0 * self.y + s1 * oy,
            z: s0 * self.z + s1 * oz,
        }
    }
}

/// Batch transform positions by view-projection matrix
#[wasm_bindgen]
pub fn transform_positions(
    positions: &[f32],
    view_proj_matrix: &[f32],
    output: &mut [f32],
) {
    let m = view_proj_matrix;
    let num_points = positions.len() / 3;

    for i in 0..num_points {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        let w = m[3] * x + m[7] * y + m[11] * z + m[15];
        let w_inv = if w.abs() > 1e-10 { 1.0 / w } else { 1.0 };

        output[i * 4] = (m[0] * x + m[4] * y + m[8] * z + m[12]) * w_inv;
        output[i * 4 + 1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) * w_inv;
        output[i * 4 + 2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) * w_inv;
        output[i * 4 + 3] = w;
    }
}
