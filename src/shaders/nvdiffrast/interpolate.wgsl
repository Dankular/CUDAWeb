// nvdiffrast-equivalent Interpolation Shader
// Attribute interpolation with perspective correction
// Supports forward pass and gradient computation for differentiable rendering

struct Vertex {
    position: vec4<f32>,
    attributes: vec4<f32>,
};

struct RasterOutput {
    triangle_id: i32,
    bary_u: f32,
    bary_v: f32,
    depth: f32,
};

struct Triangle {
    v0: u32,
    v1: u32,
    v2: u32,
    _pad: u32,
};

struct InterpolateUniforms {
    width: u32,
    height: u32,
    num_attributes: u32,
    diff_attrs: u32,  // Bitmask of which attributes have gradients
};

// Forward interpolation: compute interpolated attributes at each pixel
@group(0) @binding(0) var<storage, read> raster_output: array<RasterOutput>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> vertex_attributes: array<f32>; // Flattened: num_vertices * num_attrs
@group(0) @binding(3) var<storage, read_write> interpolated: array<f32>; // Flattened: width * height * num_attrs
@group(0) @binding(4) var<uniform> params: InterpolateUniforms;

@compute @workgroup_size(16, 16)
fn interpolate_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= params.width || pixel_y >= params.height) {
        return;
    }

    let pixel_idx = pixel_y * params.width + pixel_x;
    let raster = raster_output[pixel_idx];

    let output_base = pixel_idx * params.num_attributes;

    if (raster.triangle_id < 0) {
        // Background pixel - zero out
        for (var i = 0u; i < params.num_attributes; i++) {
            interpolated[output_base + i] = 0.0;
        }
        return;
    }

    let tri = triangles[u32(raster.triangle_id)];

    // Barycentric coordinates
    let bary_w = 1.0 - raster.bary_u - raster.bary_v;
    let bary_u = raster.bary_u;
    let bary_v = raster.bary_v;

    // Interpolate each attribute
    for (var i = 0u; i < params.num_attributes; i++) {
        let attr0 = vertex_attributes[tri.v0 * params.num_attributes + i];
        let attr1 = vertex_attributes[tri.v1 * params.num_attributes + i];
        let attr2 = vertex_attributes[tri.v2 * params.num_attributes + i];

        interpolated[output_base + i] = bary_w * attr0 + bary_u * attr1 + bary_v * attr2;
    }
}

// Backward pass: compute gradients with respect to vertex attributes
@group(0) @binding(0) var<storage, read> raster_output_grad: array<RasterOutput>;
@group(0) @binding(1) var<storage, read> triangles_grad: array<Triangle>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>; // Gradients from next layer
@group(0) @binding(3) var<storage, read_write> grad_vertex_attrs: array<atomic<u32>>; // Atomically accumulated gradients
@group(0) @binding(4) var<uniform> grad_params: InterpolateUniforms;

// Helper to atomically add float using u32 atomics
fn atomic_add_f32(ptr: ptr<storage, atomic<u32>, read_write>, value: f32) {
    var expected = atomicLoad(ptr);
    loop {
        let current_f32 = bitcast<f32>(expected);
        let new_f32 = current_f32 + value;
        let new_u32 = bitcast<u32>(new_f32);

        let result = atomicCompareExchangeWeak(ptr, expected, new_u32);
        if (result.exchanged) {
            break;
        }
        expected = result.old_value;
    }
}

@compute @workgroup_size(16, 16)
fn interpolate_backward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= grad_params.width || pixel_y >= grad_params.height) {
        return;
    }

    let pixel_idx = pixel_y * grad_params.width + pixel_x;
    let raster = raster_output_grad[pixel_idx];

    if (raster.triangle_id < 0) {
        return;
    }

    let tri = triangles_grad[u32(raster.triangle_id)];

    // Barycentric coordinates
    let bary_w = 1.0 - raster.bary_u - raster.bary_v;
    let bary_u = raster.bary_u;
    let bary_v = raster.bary_v;

    let grad_base = pixel_idx * grad_params.num_attributes;

    // Propagate gradients to vertex attributes
    for (var i = 0u; i < grad_params.num_attributes; i++) {
        let grad = grad_output[grad_base + i];

        // Gradient contribution from this pixel to each vertex
        let grad_v0 = grad * bary_w;
        let grad_v1 = grad * bary_u;
        let grad_v2 = grad * bary_v;

        // Accumulate atomically
        atomic_add_f32(&grad_vertex_attrs[tri.v0 * grad_params.num_attributes + i], grad_v0);
        atomic_add_f32(&grad_vertex_attrs[tri.v1 * grad_params.num_attributes + i], grad_v1);
        atomic_add_f32(&grad_vertex_attrs[tri.v2 * grad_params.num_attributes + i], grad_v2);
    }
}
