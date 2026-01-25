// nvdiffrast-equivalent Rasterization Shader
// Differentiable mesh rasterization for WebGPU
// Implements triangle rasterization with per-pixel derivatives

struct Vertex {
    position: vec4<f32>,  // Clip space position
    attributes: vec4<f32>, // Custom attributes (e.g., texture coords, normals)
};

struct Triangle {
    v0: u32,
    v1: u32,
    v2: u32,
    _pad: u32,
};

struct RasterizeUniforms {
    width: u32,
    height: u32,
    num_triangles: u32,
    enable_antialias: u32,
};

struct RasterOutput {
    triangle_id: i32,      // -1 if background
    bary_u: f32,           // Barycentric u
    bary_v: f32,           // Barycentric v
    depth: f32,            // Depth value
};

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> output: array<RasterOutput>;
@group(0) @binding(3) var<storage, read_write> depth_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: RasterizeUniforms;

// Convert float depth to atomic-comparable u32
fn depth_to_uint(depth: f32) -> u32 {
    // Map depth [0, 1] to uint range, larger depth = smaller uint for correct comparison
    return u32((1.0 - clamp(depth, 0.0, 1.0)) * 4294967295.0);
}

// Compute edge function for triangle rasterization
fn edge_function(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// Compute barycentric coordinates
fn compute_barycentric(
    p: vec2<f32>,
    v0: vec2<f32>,
    v1: vec2<f32>,
    v2: vec2<f32>
) -> vec3<f32> {
    let area = edge_function(v0, v1, v2);
    if (abs(area) < 1e-10) {
        return vec3<f32>(-1.0);
    }

    let w0 = edge_function(v1, v2, p);
    let w1 = edge_function(v2, v0, p);
    let w2 = edge_function(v0, v1, p);

    let inv_area = 1.0 / area;
    return vec3<f32>(w0 * inv_area, w1 * inv_area, w2 * inv_area);
}

// Main rasterization kernel
@compute @workgroup_size(16, 16)
fn rasterize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= params.width || pixel_y >= params.height) {
        return;
    }

    let pixel_idx = pixel_y * params.width + pixel_x;
    let pixel_center = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    // Initialize output
    var best_triangle: i32 = -1;
    var best_bary = vec3<f32>(0.0);
    var best_depth: f32 = 1.0;

    // Check all triangles (in practice, use BVH or tiled approach)
    for (var tri_idx = 0u; tri_idx < params.num_triangles; tri_idx++) {
        let tri = triangles[tri_idx];

        // Get vertex positions in clip space
        let v0_clip = vertices[tri.v0].position;
        let v1_clip = vertices[tri.v1].position;
        let v2_clip = vertices[tri.v2].position;

        // Perspective divide to NDC
        if (v0_clip.w <= 0.0 || v1_clip.w <= 0.0 || v2_clip.w <= 0.0) {
            continue;
        }

        let v0_ndc = v0_clip.xyz / v0_clip.w;
        let v1_ndc = v1_clip.xyz / v1_clip.w;
        let v2_ndc = v2_clip.xyz / v2_clip.w;

        // Convert to screen space
        let v0_screen = vec2<f32>(
            (v0_ndc.x * 0.5 + 0.5) * f32(params.width),
            (1.0 - (v0_ndc.y * 0.5 + 0.5)) * f32(params.height)
        );
        let v1_screen = vec2<f32>(
            (v1_ndc.x * 0.5 + 0.5) * f32(params.width),
            (1.0 - (v1_ndc.y * 0.5 + 0.5)) * f32(params.height)
        );
        let v2_screen = vec2<f32>(
            (v2_ndc.x * 0.5 + 0.5) * f32(params.width),
            (1.0 - (v2_ndc.y * 0.5 + 0.5)) * f32(params.height)
        );

        // Compute barycentric coordinates
        let bary = compute_barycentric(pixel_center, v0_screen, v1_screen, v2_screen);

        // Check if inside triangle
        if (bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0) {
            // Perspective-correct interpolation of depth
            let inv_w0 = 1.0 / v0_clip.w;
            let inv_w1 = 1.0 / v1_clip.w;
            let inv_w2 = 1.0 / v2_clip.w;

            let inv_w = bary.x * inv_w0 + bary.y * inv_w1 + bary.z * inv_w2;
            let depth = (bary.x * v0_ndc.z * inv_w0 + bary.y * v1_ndc.z * inv_w1 + bary.z * v2_ndc.z * inv_w2) / inv_w;

            // Depth test
            if (depth < best_depth && depth >= 0.0 && depth <= 1.0) {
                best_depth = depth;
                best_triangle = i32(tri_idx);

                // Perspective-correct barycentric for attribute interpolation
                let bary_persp = vec3<f32>(
                    bary.x * inv_w0,
                    bary.y * inv_w1,
                    bary.z * inv_w2
                ) / inv_w;
                best_bary = bary_persp;
            }
        }
    }

    // Write output
    output[pixel_idx] = RasterOutput(
        best_triangle,
        best_bary.y, // u = bary of v1
        best_bary.z, // v = bary of v2
        best_depth
    );

    // Update depth buffer atomically
    let depth_uint = depth_to_uint(best_depth);
    atomicMax(&depth_buffer[pixel_idx], depth_uint);
}

// Compute per-pixel derivatives of barycentric coordinates
// Used for texture filtering and gradient computation
struct BaryDerivatives {
    dudx: f32,
    dudy: f32,
    dvdx: f32,
    dvdy: f32,
};

@group(0) @binding(0) var<storage, read> raster_output: array<RasterOutput>;
@group(0) @binding(1) var<storage, read_write> derivatives: array<BaryDerivatives>;
@group(0) @binding(2) var<uniform> deriv_params: RasterizeUniforms;

@compute @workgroup_size(16, 16)
fn compute_derivatives(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= deriv_params.width || pixel_y >= deriv_params.height) {
        return;
    }

    let idx = pixel_y * deriv_params.width + pixel_x;
    let center = raster_output[idx];

    if (center.triangle_id < 0) {
        derivatives[idx] = BaryDerivatives(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Compute derivatives using finite differences
    var dudx: f32 = 0.0;
    var dudy: f32 = 0.0;
    var dvdx: f32 = 0.0;
    var dvdy: f32 = 0.0;

    // X derivative
    if (pixel_x < deriv_params.width - 1u) {
        let right_idx = idx + 1u;
        let right = raster_output[right_idx];
        if (right.triangle_id == center.triangle_id) {
            dudx = right.bary_u - center.bary_u;
            dvdx = right.bary_v - center.bary_v;
        }
    }

    // Y derivative
    if (pixel_y < deriv_params.height - 1u) {
        let down_idx = idx + deriv_params.width;
        let down = raster_output[down_idx];
        if (down.triangle_id == center.triangle_id) {
            dudy = down.bary_u - center.bary_u;
            dvdy = down.bary_v - center.bary_v;
        }
    }

    derivatives[idx] = BaryDerivatives(dudx, dudy, dvdx, dvdy);
}
