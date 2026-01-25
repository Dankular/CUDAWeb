// nvdiffrast-equivalent Anti-aliasing Shader
// Edge-aware anti-aliasing for differentiable rendering
// Implements both forward and backward passes

struct RasterOutput {
    triangle_id: i32,
    bary_u: f32,
    bary_v: f32,
    depth: f32,
};

struct Vertex {
    position: vec4<f32>,
    attributes: vec4<f32>,
};

struct Triangle {
    v0: u32,
    v1: u32,
    v2: u32,
    _pad: u32,
};

struct AntialiasUniforms {
    width: u32,
    height: u32,
    num_triangles: u32,
    num_layers: u32,  // For order-independent transparency
};

// Edge detection for silhouette anti-aliasing
// Returns coverage value [0, 1] for pixels near edges

@group(0) @binding(0) var<storage, read> raster_output: array<RasterOutput>;
@group(0) @binding(1) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> aa_output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: AntialiasUniforms;

// Compute signed distance to triangle edge
fn edge_distance(
    pixel: vec2<f32>,
    v0: vec2<f32>,
    v1: vec2<f32>
) -> f32 {
    let edge = v1 - v0;
    let edge_len = length(edge);
    if (edge_len < 1e-6) {
        return 1e10;
    }

    let edge_norm = edge / edge_len;
    let perp = vec2<f32>(-edge_norm.y, edge_norm.x);

    return dot(pixel - v0, perp);
}

// Compute coverage for a pixel
fn compute_coverage(
    pixel: vec2<f32>,
    raster: RasterOutput,
    vertices: ptr<storage, array<Vertex>, read>,
    triangles: ptr<storage, array<Triangle>, read>
) -> f32 {
    if (raster.triangle_id < 0) {
        return 0.0;
    }

    let tri = (*triangles)[u32(raster.triangle_id)];

    let v0_clip = (*vertices)[tri.v0].position;
    let v1_clip = (*vertices)[tri.v1].position;
    let v2_clip = (*vertices)[tri.v2].position;

    // Convert to screen space
    let v0_screen = vec2<f32>(
        (v0_clip.x / v0_clip.w * 0.5 + 0.5) * f32(params.width),
        (1.0 - (v0_clip.y / v0_clip.w * 0.5 + 0.5)) * f32(params.height)
    );
    let v1_screen = vec2<f32>(
        (v1_clip.x / v1_clip.w * 0.5 + 0.5) * f32(params.width),
        (1.0 - (v1_clip.y / v1_clip.w * 0.5 + 0.5)) * f32(params.height)
    );
    let v2_screen = vec2<f32>(
        (v2_clip.x / v2_clip.w * 0.5 + 0.5) * f32(params.width),
        (1.0 - (v2_clip.y / v2_clip.w * 0.5 + 0.5)) * f32(params.height)
    );

    // Find minimum distance to any edge
    let d0 = edge_distance(pixel, v0_screen, v1_screen);
    let d1 = edge_distance(pixel, v1_screen, v2_screen);
    let d2 = edge_distance(pixel, v2_screen, v0_screen);

    let min_dist = min(d0, min(d1, d2));

    // Convert distance to coverage using smooth step
    // Pixels within 0.5 units of edge get partial coverage
    return smoothstep(-0.5, 0.5, min_dist);
}

// Simple box-filter anti-aliasing
@group(0) @binding(0) var<storage, read> raster_input: array<RasterOutput>;
@group(0) @binding(1) var<storage, read> color_input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> aa_colors: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> aa_params: AntialiasUniforms;

@compute @workgroup_size(16, 16)
fn antialias_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= aa_params.width || pixel_y >= aa_params.height) {
        return;
    }

    let pixel_idx = pixel_y * aa_params.width + pixel_x;
    let center_raster = raster_input[pixel_idx];
    let center_color = color_input[pixel_idx];
    let center_tri = center_raster.triangle_id;

    // Check if this pixel is near a silhouette edge
    var is_edge = false;

    // Check 4-neighborhood for different triangle IDs
    if (pixel_x > 0u) {
        let left_idx = pixel_idx - 1u;
        if (raster_input[left_idx].triangle_id != center_tri) {
            is_edge = true;
        }
    }
    if (pixel_x < aa_params.width - 1u) {
        let right_idx = pixel_idx + 1u;
        if (raster_input[right_idx].triangle_id != center_tri) {
            is_edge = true;
        }
    }
    if (pixel_y > 0u) {
        let top_idx = pixel_idx - aa_params.width;
        if (raster_input[top_idx].triangle_id != center_tri) {
            is_edge = true;
        }
    }
    if (pixel_y < aa_params.height - 1u) {
        let bottom_idx = pixel_idx + aa_params.width;
        if (raster_input[bottom_idx].triangle_id != center_tri) {
            is_edge = true;
        }
    }

    if (!is_edge) {
        // Not an edge pixel - pass through
        aa_colors[pixel_idx] = center_color;
        return;
    }

    // Edge pixel - blend with neighbors
    var sum_color = center_color;
    var count = 1.0;

    // 3x3 box filter for edge pixels
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            let nx = i32(pixel_x) + dx;
            let ny = i32(pixel_y) + dy;

            if (nx >= 0 && nx < i32(aa_params.width) && ny >= 0 && ny < i32(aa_params.height)) {
                let neighbor_idx = u32(ny) * aa_params.width + u32(nx);
                sum_color += color_input[neighbor_idx];
                count += 1.0;
            }
        }
    }

    aa_colors[pixel_idx] = sum_color / count;
}

// FXAA-style edge-aware anti-aliasing
@compute @workgroup_size(16, 16)
fn antialias_fxaa(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= aa_params.width || pixel_y >= aa_params.height) {
        return;
    }

    let pixel_idx = pixel_y * aa_params.width + pixel_x;

    // Get luminance of center and neighbors
    let center = color_input[pixel_idx];
    let luma_center = dot(center.rgb, vec3<f32>(0.299, 0.587, 0.114));

    var luma_n = luma_center;
    var luma_s = luma_center;
    var luma_e = luma_center;
    var luma_w = luma_center;

    if (pixel_y > 0u) {
        let n = color_input[pixel_idx - aa_params.width];
        luma_n = dot(n.rgb, vec3<f32>(0.299, 0.587, 0.114));
    }
    if (pixel_y < aa_params.height - 1u) {
        let s = color_input[pixel_idx + aa_params.width];
        luma_s = dot(s.rgb, vec3<f32>(0.299, 0.587, 0.114));
    }
    if (pixel_x < aa_params.width - 1u) {
        let e = color_input[pixel_idx + 1u];
        luma_e = dot(e.rgb, vec3<f32>(0.299, 0.587, 0.114));
    }
    if (pixel_x > 0u) {
        let w = color_input[pixel_idx - 1u];
        luma_w = dot(w.rgb, vec3<f32>(0.299, 0.587, 0.114));
    }

    // Compute contrast
    let luma_max = max(luma_center, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    let luma_min = min(luma_center, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let luma_range = luma_max - luma_min;

    // Skip if contrast is low
    let FXAA_EDGE_THRESHOLD = 0.125;
    let FXAA_EDGE_THRESHOLD_MIN = 0.0312;

    if (luma_range < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * FXAA_EDGE_THRESHOLD)) {
        aa_colors[pixel_idx] = center;
        return;
    }

    // Determine edge direction
    let edge_h = abs(luma_n + luma_s - 2.0 * luma_center);
    let edge_v = abs(luma_e + luma_w - 2.0 * luma_center);
    let is_horizontal = edge_h > edge_v;

    // Blend along edge
    var blend: f32;
    if (is_horizontal) {
        blend = abs(luma_n - luma_s) / luma_range;
    } else {
        blend = abs(luma_e - luma_w) / luma_range;
    }

    blend = smoothstep(0.0, 1.0, blend);
    blend = blend * blend;

    // Apply blend
    var result = center;
    if (is_horizontal) {
        if (pixel_y > 0u && pixel_y < aa_params.height - 1u) {
            let n_color = color_input[pixel_idx - aa_params.width];
            let s_color = color_input[pixel_idx + aa_params.width];
            result = mix(center, (n_color + s_color) * 0.5, blend * 0.5);
        }
    } else {
        if (pixel_x > 0u && pixel_x < aa_params.width - 1u) {
            let e_color = color_input[pixel_idx + 1u];
            let w_color = color_input[pixel_idx - 1u];
            result = mix(center, (e_color + w_color) * 0.5, blend * 0.5);
        }
    }

    aa_colors[pixel_idx] = result;
}

// Backward pass for differentiable anti-aliasing
// Distributes gradients from AA output back to original colors
@group(0) @binding(0) var<storage, read> grad_aa_output: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> raster_backward: array<RasterOutput>;
@group(0) @binding(2) var<storage, read_write> grad_colors: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> grad_aa_params: AntialiasUniforms;

@compute @workgroup_size(16, 16)
fn antialias_backward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= grad_aa_params.width || pixel_y >= grad_aa_params.height) {
        return;
    }

    let pixel_idx = pixel_y * grad_aa_params.width + pixel_x;
    let grad = grad_aa_output[pixel_idx];
    let center_tri = raster_backward[pixel_idx].triangle_id;

    // Check if this was an edge pixel (same logic as forward)
    var is_edge = false;

    if (pixel_x > 0u && raster_backward[pixel_idx - 1u].triangle_id != center_tri) { is_edge = true; }
    if (pixel_x < grad_aa_params.width - 1u && raster_backward[pixel_idx + 1u].triangle_id != center_tri) { is_edge = true; }
    if (pixel_y > 0u && raster_backward[pixel_idx - grad_aa_params.width].triangle_id != center_tri) { is_edge = true; }
    if (pixel_y < grad_aa_params.height - 1u && raster_backward[pixel_idx + grad_aa_params.width].triangle_id != center_tri) { is_edge = true; }

    if (!is_edge) {
        // Not an edge pixel - gradient passes through directly
        grad_colors[pixel_idx] = grad;
        return;
    }

    // Edge pixel - distribute gradient to all pixels that contributed to the blend
    // For box filter, each pixel in 3x3 region gets 1/9 of the gradient
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = i32(pixel_x) + dx;
            let ny = i32(pixel_y) + dy;
            if (nx >= 0 && nx < i32(grad_aa_params.width) && ny >= 0 && ny < i32(grad_aa_params.height)) {
                count += 1.0;
            }
        }
    }

    let weight = 1.0 / count;

    // This pixel's gradient contribution
    grad_colors[pixel_idx] = grad * weight;

    // Note: In a full implementation, you'd also need to atomically add gradients
    // to neighboring pixels that contributed to this pixel's blended value
}
