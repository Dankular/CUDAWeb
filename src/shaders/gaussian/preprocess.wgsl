// Gaussian Splatting Preprocessing Shader
// Transforms 3D Gaussians to view space and computes 2D projections
// Equivalent to CUDA preprocess kernel

struct Gaussian3D {
    position: vec3<f32>,    // 3D position (xyz)
    opacity: f32,           // Opacity (sigmoid-activated)
    scale: vec3<f32>,       // Scale in 3D
    _pad1: f32,
    rotation: vec4<f32>,    // Rotation quaternion (wxyz)
};

struct Gaussian2D {
    position: vec2<f32>,    // 2D screen position
    depth: f32,             // Depth for sorting
    radius: f32,            // Bounding radius in pixels
    conic: vec3<f32>,       // Conic coefficients (a, b, c)
    opacity: f32,           // Final opacity
    color: vec4<f32>,       // SH-evaluated color (rgb) + padding
};

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_proj_matrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    focal_x: f32,
    focal_y: f32,
    tan_fov_x: f32,
    tan_fov_y: f32,
    width: f32,
    height: f32,
    near: f32,
    far: f32,
    _pad: f32,
};

struct PreprocessUniforms {
    num_gaussians: u32,
    sh_degree: u32,
    scale_modifier: f32,
    _pad: u32,
};

// Spherical Harmonics coefficients (up to degree 3 = 16 coefficients per channel)
struct SHCoefficients {
    coeffs: array<vec3<f32>, 16>,
};

@group(0) @binding(0) var<storage, read> gaussians_3d: array<Gaussian3D>;
@group(0) @binding(1) var<storage, read> sh_coefficients: array<SHCoefficients>;
@group(0) @binding(2) var<storage, read_write> gaussians_2d: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> visible_count: atomic<u32>;
@group(0) @binding(5) var<uniform> camera: CameraUniforms;
@group(0) @binding(6) var<uniform> params: PreprocessUniforms;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

// Convert quaternion to rotation matrix
fn quat_to_matrix(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx2 = x * x2;
    let yy2 = y * y2;
    let zz2 = z * z2;
    let xy2 = x * y2;
    let xz2 = x * z2;
    let yz2 = y * z2;
    let wx2 = w * x2;
    let wy2 = w * y2;
    let wz2 = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - yy2 - zz2, xy2 + wz2, xz2 - wy2),
        vec3<f32>(xy2 - wz2, 1.0 - xx2 - zz2, yz2 + wx2),
        vec3<f32>(xz2 + wy2, yz2 - wx2, 1.0 - xx2 - yy2)
    );
}

// Compute 3D covariance matrix from scale and rotation
fn compute_covariance_3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    let R = quat_to_matrix(rotation);
    let S = mat3x3<f32>(
        vec3<f32>(scale.x * scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y * scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z * scale.z)
    );
    // Cov = R * S * S * R^T
    return R * S * transpose(R);
}

// Project 3D covariance to 2D
fn compute_covariance_2d(
    mean_3d: vec3<f32>,
    cov_3d: mat3x3<f32>,
    view_matrix: mat4x4<f32>,
    focal_x: f32,
    focal_y: f32
) -> mat2x2<f32> {
    // Transform mean to view space
    let t = (view_matrix * vec4<f32>(mean_3d, 1.0)).xyz;

    let limx = 1.3 * camera.tan_fov_x;
    let limy = 1.3 * camera.tan_fov_y;
    let tx = min(limx, max(-limx, t.x / t.z)) * t.z;
    let ty = min(limy, max(-limy, t.y / t.z)) * t.z;

    // Jacobian of perspective projection
    let J = mat3x3<f32>(
        vec3<f32>(focal_x / t.z, 0.0, 0.0),
        vec3<f32>(0.0, focal_y / t.z, 0.0),
        vec3<f32>(-focal_x * tx / (t.z * t.z), -focal_y * ty / (t.z * t.z), 0.0)
    );

    // View matrix rotation part
    let W = mat3x3<f32>(
        view_matrix[0].xyz,
        view_matrix[1].xyz,
        view_matrix[2].xyz
    );

    let T = J * W;
    let cov_2d = T * cov_3d * transpose(T);

    // Add low-pass filter for anti-aliasing
    return mat2x2<f32>(
        vec2<f32>(cov_2d[0][0] + 0.3, cov_2d[0][1]),
        vec2<f32>(cov_2d[1][0], cov_2d[1][1] + 0.3)
    );
}

// Compute inverse of 2D covariance (conic)
fn compute_conic(cov_2d: mat2x2<f32>) -> vec3<f32> {
    let det = cov_2d[0][0] * cov_2d[1][1] - cov_2d[0][1] * cov_2d[1][0];
    if (det == 0.0) {
        return vec3<f32>(0.0);
    }
    let det_inv = 1.0 / det;
    return vec3<f32>(
        cov_2d[1][1] * det_inv,
        -cov_2d[0][1] * det_inv,
        cov_2d[0][0] * det_inv
    );
}

// Evaluate spherical harmonics
fn evaluate_sh(
    sh: SHCoefficients,
    dir: vec3<f32>,
    degree: u32
) -> vec3<f32> {
    var result = SH_C0 * sh.coeffs[0];

    if (degree > 0u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh.coeffs[1] + z * sh.coeffs[2] - x * sh.coeffs[3]);

        if (degree > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;

            result = result +
                SH_C2_0 * xy * sh.coeffs[4] +
                SH_C2_1 * yz * sh.coeffs[5] +
                SH_C2_2 * (2.0 * zz - xx - yy) * sh.coeffs[6] +
                SH_C2_3 * xz * sh.coeffs[7] +
                SH_C2_4 * (xx - yy) * sh.coeffs[8];

            if (degree > 2u) {
                result = result +
                    SH_C3_0 * y * (3.0 * xx - yy) * sh.coeffs[9] +
                    SH_C3_1 * xy * z * sh.coeffs[10] +
                    SH_C3_2 * y * (4.0 * zz - xx - yy) * sh.coeffs[11] +
                    SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh.coeffs[12] +
                    SH_C3_4 * x * (4.0 * zz - xx - yy) * sh.coeffs[13] +
                    SH_C3_5 * z * (xx - yy) * sh.coeffs[14] +
                    SH_C3_6 * x * (xx - 3.0 * yy) * sh.coeffs[15];
            }
        }
    }

    return max(result + vec3<f32>(0.5), vec3<f32>(0.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_gaussians) {
        return;
    }

    let gaussian = gaussians_3d[idx];

    // Transform to clip space
    let pos_world = vec4<f32>(gaussian.position, 1.0);
    let pos_view = camera.view_matrix * pos_world;
    let pos_clip = camera.proj_matrix * pos_view;

    // Frustum culling
    let w = pos_clip.w;
    if (w <= 0.0) {
        return;
    }

    let pos_ndc = pos_clip.xyz / w;

    // Check if in view frustum with some margin
    if (pos_ndc.x < -1.3 || pos_ndc.x > 1.3 ||
        pos_ndc.y < -1.3 || pos_ndc.y > 1.3 ||
        pos_ndc.z < 0.0 || pos_ndc.z > 1.0) {
        return;
    }

    // Compute screen space position
    let screen_x = (pos_ndc.x * 0.5 + 0.5) * camera.width;
    let screen_y = (1.0 - (pos_ndc.y * 0.5 + 0.5)) * camera.height;

    // Compute 3D covariance
    let scale = gaussian.scale * params.scale_modifier;
    let cov_3d = compute_covariance_3d(scale, gaussian.rotation);

    // Project to 2D covariance
    let cov_2d = compute_covariance_2d(
        gaussian.position,
        cov_3d,
        camera.view_matrix,
        camera.focal_x,
        camera.focal_y
    );

    // Compute conic (inverse of 2D covariance)
    let conic = compute_conic(cov_2d);
    if (conic.x == 0.0 && conic.y == 0.0 && conic.z == 0.0) {
        return;
    }

    // Compute bounding radius (3 sigma)
    let det = cov_2d[0][0] * cov_2d[1][1] - cov_2d[0][1] * cov_2d[1][0];
    let mid = 0.5 * (cov_2d[0][0] + cov_2d[1][1]);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Skip if too small
    if (radius < 1.0) {
        return;
    }

    // Evaluate spherical harmonics for color
    let view_dir = normalize(gaussian.position - camera.camera_position);
    let sh = sh_coefficients[idx];
    let color = evaluate_sh(sh, view_dir, params.sh_degree);

    // Apply sigmoid to opacity
    let opacity = 1.0 / (1.0 + exp(-gaussian.opacity));

    // Allocate output slot
    let out_idx = atomicAdd(&visible_count, 1u);

    // Write output
    visible_indices[out_idx] = idx;
    gaussians_2d[out_idx] = Gaussian2D(
        vec2<f32>(screen_x, screen_y),
        pos_view.z,
        radius,
        conic,
        opacity,
        vec4<f32>(color, 1.0)
    );
}
