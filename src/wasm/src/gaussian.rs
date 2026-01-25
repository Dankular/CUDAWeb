//! Gaussian Splatting data structures and operations

use wasm_bindgen::prelude::*;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

/// Represents a single 3D Gaussian
#[derive(Clone, Debug)]
pub struct Gaussian3D {
    pub position: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub opacity: f32,
    pub sh_coefficients: Vec<Vector3<f32>>,
}

impl Gaussian3D {
    pub fn new(
        position: Vector3<f32>,
        scale: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
        opacity: f32,
    ) -> Self {
        Self {
            position,
            scale,
            rotation,
            opacity,
            sh_coefficients: vec![Vector3::new(0.5, 0.5, 0.5)], // Default gray DC
        }
    }

    /// Compute the 3D covariance matrix
    pub fn compute_covariance(&self) -> Matrix3<f32> {
        let r = self.rotation.to_rotation_matrix();
        let s = Matrix3::from_diagonal(&Vector3::new(
            self.scale.x * self.scale.x,
            self.scale.y * self.scale.y,
            self.scale.z * self.scale.z,
        ));
        r.matrix() * s * r.matrix().transpose()
    }
}

/// Gaussian data processor for CPU operations
#[wasm_bindgen]
pub struct GaussianProcessor {
    num_gaussians: usize,
}

#[wasm_bindgen]
impl GaussianProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(num_gaussians: usize) -> Self {
        Self { num_gaussians }
    }

    /// Normalize quaternion rotations in-place
    #[wasm_bindgen]
    pub fn normalize_rotations(&self, rotations: &mut [f32]) {
        for i in (0..rotations.len()).step_by(4) {
            let w = rotations[i];
            let x = rotations[i + 1];
            let y = rotations[i + 2];
            let z = rotations[i + 3];

            let len = (w * w + x * x + y * y + z * z).sqrt();
            if len > 0.0 {
                rotations[i] = w / len;
                rotations[i + 1] = x / len;
                rotations[i + 2] = y / len;
                rotations[i + 3] = z / len;
            }
        }
    }

    /// Apply activation functions to raw Gaussian parameters
    /// - scales: exp(scale) for positive values
    /// - opacities: sigmoid(opacity) for [0, 1] range
    #[wasm_bindgen]
    pub fn activate_parameters(&self, scales: &mut [f32], opacities: &mut [f32]) {
        // Activate scales with exp
        for scale in scales.iter_mut() {
            *scale = scale.exp();
        }

        // Activate opacities with sigmoid
        for opacity in opacities.iter_mut() {
            *opacity = 1.0 / (1.0 + (-*opacity).exp());
        }
    }

    /// Compute SH coefficients for view-dependent color
    /// Returns RGB color for each Gaussian given a view direction
    #[wasm_bindgen]
    pub fn evaluate_sh(
        &self,
        sh_coefficients: &[f32],
        view_dir_x: f32,
        view_dir_y: f32,
        view_dir_z: f32,
        sh_degree: u32,
        output_colors: &mut [f32],
    ) {
        let num_sh_coeffs = ((sh_degree + 1) * (sh_degree + 1)) as usize;
        let dir = Vector3::new(view_dir_x, view_dir_y, view_dir_z).normalize();

        // SH constants
        const SH_C0: f32 = 0.28209479177387814;
        const SH_C1: f32 = 0.4886025119029199;
        const SH_C2: [f32; 5] = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ];
        const SH_C3: [f32; 7] = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        ];

        for i in 0..self.num_gaussians {
            let base = i * num_sh_coeffs * 3;
            let out_base = i * 3;

            // DC term (degree 0)
            let mut color = Vector3::new(
                SH_C0 * sh_coefficients[base],
                SH_C0 * sh_coefficients[base + 1],
                SH_C0 * sh_coefficients[base + 2],
            );

            if sh_degree >= 1 && num_sh_coeffs >= 4 {
                // Degree 1
                let x = dir.x;
                let y = dir.y;
                let z = dir.z;

                color += SH_C1
                    * Vector3::new(
                        -y * sh_coefficients[base + 3]
                            + z * sh_coefficients[base + 6]
                            - x * sh_coefficients[base + 9],
                        -y * sh_coefficients[base + 4]
                            + z * sh_coefficients[base + 7]
                            - x * sh_coefficients[base + 10],
                        -y * sh_coefficients[base + 5]
                            + z * sh_coefficients[base + 8]
                            - x * sh_coefficients[base + 11],
                    );

                if sh_degree >= 2 && num_sh_coeffs >= 9 {
                    // Degree 2
                    let xx = x * x;
                    let yy = y * y;
                    let zz = z * z;
                    let xy = x * y;
                    let xz = x * z;
                    let yz = y * z;

                    let sh_base = base + 12;
                    color += Vector3::new(
                        SH_C2[0] * xy * sh_coefficients[sh_base]
                            + SH_C2[1] * yz * sh_coefficients[sh_base + 3]
                            + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coefficients[sh_base + 6]
                            + SH_C2[3] * xz * sh_coefficients[sh_base + 9]
                            + SH_C2[4] * (xx - yy) * sh_coefficients[sh_base + 12],
                        SH_C2[0] * xy * sh_coefficients[sh_base + 1]
                            + SH_C2[1] * yz * sh_coefficients[sh_base + 4]
                            + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coefficients[sh_base + 7]
                            + SH_C2[3] * xz * sh_coefficients[sh_base + 10]
                            + SH_C2[4] * (xx - yy) * sh_coefficients[sh_base + 13],
                        SH_C2[0] * xy * sh_coefficients[sh_base + 2]
                            + SH_C2[1] * yz * sh_coefficients[sh_base + 5]
                            + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coefficients[sh_base + 8]
                            + SH_C2[3] * xz * sh_coefficients[sh_base + 11]
                            + SH_C2[4] * (xx - yy) * sh_coefficients[sh_base + 14],
                    );
                }
            }

            // Clamp to valid color range
            output_colors[out_base] = (color.x + 0.5).max(0.0);
            output_colors[out_base + 1] = (color.y + 0.5).max(0.0);
            output_colors[out_base + 2] = (color.z + 0.5).max(0.0);
        }
    }

    /// Filter Gaussians by opacity threshold
    #[wasm_bindgen]
    pub fn filter_by_opacity(&self, opacities: &[f32], threshold: f32) -> Vec<u32> {
        opacities
            .iter()
            .enumerate()
            .filter(|(_, &op)| op >= threshold)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Compute distance from camera for each Gaussian
    #[wasm_bindgen]
    pub fn compute_distances(
        &self,
        positions: &[f32],
        camera_x: f32,
        camera_y: f32,
        camera_z: f32,
        distances: &mut [f32],
    ) {
        let camera = Vector3::new(camera_x, camera_y, camera_z);

        for i in 0..self.num_gaussians {
            let pos = Vector3::new(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
            );
            distances[i] = (pos - camera).norm_squared();
        }
    }

    /// Cull Gaussians outside view frustum
    #[wasm_bindgen]
    pub fn frustum_cull(
        &self,
        positions: &[f32],
        view_proj_matrix: &[f32], // 16 floats (4x4 matrix)
        output_visible: &mut [u32],
    ) -> usize {
        let mut visible_count = 0;

        for i in 0..self.num_gaussians {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            // Transform to clip space
            let clip_x = view_proj_matrix[0] * x
                + view_proj_matrix[4] * y
                + view_proj_matrix[8] * z
                + view_proj_matrix[12];
            let clip_y = view_proj_matrix[1] * x
                + view_proj_matrix[5] * y
                + view_proj_matrix[9] * z
                + view_proj_matrix[13];
            let clip_z = view_proj_matrix[2] * x
                + view_proj_matrix[6] * y
                + view_proj_matrix[10] * z
                + view_proj_matrix[14];
            let clip_w = view_proj_matrix[3] * x
                + view_proj_matrix[7] * y
                + view_proj_matrix[11] * z
                + view_proj_matrix[15];

            // Check if inside frustum (with small margin)
            if clip_w > 0.0
                && clip_x >= -clip_w * 1.2
                && clip_x <= clip_w * 1.2
                && clip_y >= -clip_w * 1.2
                && clip_y <= clip_w * 1.2
                && clip_z >= 0.0
                && clip_z <= clip_w
            {
                output_visible[visible_count] = i as u32;
                visible_count += 1;
            }
        }

        visible_count
    }
}

/// Pack Gaussian data into interleaved format for GPU upload
#[wasm_bindgen]
pub fn pack_gaussians(
    positions: &[f32],
    scales: &[f32],
    rotations: &[f32],
    opacities: &[f32],
) -> Vec<f32> {
    let num_gaussians = positions.len() / 3;

    // Packed format: [pos.xyz, opacity, scale.xyz, pad, rot.xyzw]
    // = 12 floats per Gaussian
    let mut packed = vec![0.0f32; num_gaussians * 12];

    for i in 0..num_gaussians {
        let base = i * 12;

        // Position (3)
        packed[base] = positions[i * 3];
        packed[base + 1] = positions[i * 3 + 1];
        packed[base + 2] = positions[i * 3 + 2];

        // Opacity (1)
        packed[base + 3] = opacities[i];

        // Scale (3)
        packed[base + 4] = scales[i * 3];
        packed[base + 5] = scales[i * 3 + 1];
        packed[base + 6] = scales[i * 3 + 2];

        // Padding (1)
        packed[base + 7] = 0.0;

        // Rotation quaternion (4)
        packed[base + 8] = rotations[i * 4];
        packed[base + 9] = rotations[i * 4 + 1];
        packed[base + 10] = rotations[i * 4 + 2];
        packed[base + 11] = rotations[i * 4 + 3];
    }

    packed
}
