// Gaussian Splatting Rasterization Shader
// Tile-based alpha blending of sorted Gaussians
// Equivalent to CUDA rasterization kernel

struct Gaussian2D {
    position: vec2<f32>,
    depth: f32,
    radius: f32,
    conic: vec3<f32>,
    opacity: f32,
    color: vec4<f32>,
};

struct RasterUniforms {
    tile_size: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    width: u32,
    height: u32,
    background_color: vec3<f32>,
    _pad: u32,
};

struct TileRange {
    start: u32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> gaussians_2d: array<Gaussian2D>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read> tile_ranges: array<TileRange>;
@group(0) @binding(3) var<uniform> params: RasterUniforms;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Shared memory for tile processing
var<workgroup> shared_gaussians: array<Gaussian2D, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// Evaluate Gaussian at pixel position
fn evaluate_gaussian(
    pixel: vec2<f32>,
    gaussian: Gaussian2D
) -> f32 {
    let d = pixel - gaussian.position;
    let conic = gaussian.conic;

    // Compute exponent: -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
    let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);

    // Early exit if too far from center
    if (power < -4.0) {
        return 0.0;
    }

    return exp(power);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;
    let local_idx = local_id.y * 16u + local_id.x;

    // Check bounds
    if (pixel_x >= params.width || pixel_y >= params.height) {
        return;
    }

    // Get tile ID
    let tile_x = workgroup_id.x;
    let tile_y = workgroup_id.y;
    let tile_id = tile_y * params.num_tiles_x + tile_x;

    // Get range of Gaussians for this tile
    let tile_range = tile_ranges[tile_id];
    let pixel = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    // Initialize accumulation
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var T = 1.0;  // Transmittance

    // Process Gaussians in batches
    let batch_size = 256u;
    var gaussian_idx = tile_range.start;
    let end_idx = tile_range.start + tile_range.count;

    while (gaussian_idx < end_idx && T > 0.001) {
        // Load batch into shared memory
        let batch_end = min(gaussian_idx + batch_size, end_idx);
        let batch_count = batch_end - gaussian_idx;

        if (local_idx < batch_count) {
            let sorted_idx = sorted_indices[gaussian_idx + local_idx];
            shared_indices[local_idx] = sorted_idx;
            shared_gaussians[local_idx] = gaussians_2d[sorted_idx];
        }
        workgroupBarrier();

        // Process batch
        for (var i = 0u; i < batch_count; i = i + 1u) {
            let gaussian = shared_gaussians[i];

            // Early culling: check if pixel is within radius
            let d = abs(pixel - gaussian.position);
            if (d.x > gaussian.radius || d.y > gaussian.radius) {
                continue;
            }

            // Evaluate Gaussian
            let G = evaluate_gaussian(pixel, gaussian);
            if (G < 0.001) {
                continue;
            }

            let alpha = min(0.99, gaussian.opacity * G);

            // Skip if alpha is negligible
            if (alpha < 1.0 / 255.0) {
                continue;
            }

            // Front-to-back compositing
            let weight = alpha * T;
            accumulated_color = accumulated_color + weight * gaussian.color.rgb;
            accumulated_alpha = accumulated_alpha + weight;
            T = T * (1.0 - alpha);

            // Early termination if fully opaque
            if (T < 0.001) {
                break;
            }
        }

        workgroupBarrier();
        gaussian_idx = batch_end;
    }

    // Composite with background
    let final_color = accumulated_color + T * params.background_color;

    // Write to output
    textureStore(output_texture, vec2<i32>(i32(pixel_x), i32(pixel_y)), vec4<f32>(final_color, 1.0));
}

// Alternative: Render to vertex buffer for hardware rasterization
// This generates quads for each Gaussian that can be rendered with standard pipeline

struct GaussianVertex {
    position: vec4<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>,
    opacity: f32,
};

@group(0) @binding(0) var<storage, read> gaussians_2d_input: array<Gaussian2D>;
@group(0) @binding(1) var<storage, read_write> vertices: array<GaussianVertex>;
@group(0) @binding(2) var<uniform> vertex_params: RasterUniforms;

@compute @workgroup_size(256)
fn generate_vertices(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let gaussian = gaussians_2d_input[idx];

    // Skip invalid Gaussians
    if (gaussian.opacity <= 0.0) {
        return;
    }

    let center = gaussian.position;
    let radius = gaussian.radius;

    // Generate 4 vertices for a quad (6 for 2 triangles with indices)
    let base_vertex = idx * 4u;

    // Convert to NDC
    let width = f32(vertex_params.width);
    let height = f32(vertex_params.height);

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    for (var i = 0u; i < 4u; i = i + 1u) {
        let screen_pos = center + offsets[i] * radius;
        let ndc_x = (screen_pos.x / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.y / height) * 2.0;

        vertices[base_vertex + i] = GaussianVertex(
            vec4<f32>(ndc_x, ndc_y, gaussian.depth, 1.0),
            uvs[i],
            gaussian.color,
            gaussian.conic,
            gaussian.opacity
        );
    }
}
