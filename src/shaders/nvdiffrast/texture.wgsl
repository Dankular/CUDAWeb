// nvdiffrast-equivalent Texture Sampling Shader
// Differentiable texture lookup with mip-mapping support
// Supports forward and backward passes for gradient computation

struct TextureUniforms {
    width: u32,
    height: u32,
    tex_width: u32,
    tex_height: u32,
    num_mips: u32,
    filter_mode: u32,  // 0 = nearest, 1 = bilinear, 2 = trilinear
    boundary_mode: u32, // 0 = clamp, 1 = repeat, 2 = mirror
    _pad: u32,
};

struct BaryDerivatives {
    dudx: f32,
    dudy: f32,
    dvdx: f32,
    dvdy: f32,
};

// Texture coordinates per pixel
@group(0) @binding(0) var<storage, read> tex_coords: array<vec2<f32>>;
// Derivatives for mip selection
@group(0) @binding(1) var<storage, read> derivatives: array<BaryDerivatives>;
// Texture data (flattened mip chain)
@group(0) @binding(2) var<storage, read> texture_data: array<vec4<f32>>;
// Mip level offsets
@group(0) @binding(3) var<storage, read> mip_offsets: array<u32>;
// Output sampled colors
@group(0) @binding(4) var<storage, read_write> output_colors: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: TextureUniforms;

// Apply boundary mode to texture coordinate
fn apply_boundary(coord: f32, size: f32, mode: u32) -> f32 {
    switch (mode) {
        case 0u: { // Clamp
            return clamp(coord, 0.0, size - 1.0);
        }
        case 1u: { // Repeat
            var c = coord % size;
            if (c < 0.0) { c += size; }
            return c;
        }
        case 2u: { // Mirror
            let period = size * 2.0;
            var c = abs(coord % period);
            if (c >= size) { c = period - c; }
            return clamp(c, 0.0, size - 1.0);
        }
        default: {
            return clamp(coord, 0.0, size - 1.0);
        }
    }
}

// Sample texture at specific mip level with bilinear interpolation
fn sample_mip_bilinear(
    uv: vec2<f32>,
    mip_level: u32,
    mip_width: u32,
    mip_height: u32,
    mip_offset: u32
) -> vec4<f32> {
    let tex_x = uv.x * f32(mip_width) - 0.5;
    let tex_y = uv.y * f32(mip_height) - 0.5;

    let x0 = u32(floor(apply_boundary(tex_x, f32(mip_width), params.boundary_mode)));
    let y0 = u32(floor(apply_boundary(tex_y, f32(mip_height), params.boundary_mode)));
    let x1 = u32(floor(apply_boundary(tex_x + 1.0, f32(mip_width), params.boundary_mode)));
    let y1 = u32(floor(apply_boundary(tex_y + 1.0, f32(mip_height), params.boundary_mode)));

    let fx = fract(tex_x);
    let fy = fract(tex_y);

    let idx00 = mip_offset + y0 * mip_width + x0;
    let idx10 = mip_offset + y0 * mip_width + x1;
    let idx01 = mip_offset + y1 * mip_width + x0;
    let idx11 = mip_offset + y1 * mip_width + x1;

    let c00 = texture_data[idx00];
    let c10 = texture_data[idx10];
    let c01 = texture_data[idx01];
    let c11 = texture_data[idx11];

    let c0 = mix(c00, c10, fx);
    let c1 = mix(c01, c11, fx);

    return mix(c0, c1, fy);
}

// Compute mip level from derivatives
fn compute_mip_level(deriv: BaryDerivatives, tex_width: f32, tex_height: f32) -> f32 {
    // Scale derivatives by texture size
    let duvdx = vec2<f32>(deriv.dudx * tex_width, deriv.dvdx * tex_height);
    let duvdy = vec2<f32>(deriv.dudy * tex_width, deriv.dvdy * tex_height);

    // Compute maximum rate of change
    let max_deriv = max(dot(duvdx, duvdx), dot(duvdy, duvdy));

    // Mip level = log2(sqrt(max_deriv)) = 0.5 * log2(max_deriv)
    return 0.5 * log2(max(max_deriv, 1.0));
}

@compute @workgroup_size(16, 16)
fn texture_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= params.width || pixel_y >= params.height) {
        return;
    }

    let pixel_idx = pixel_y * params.width + pixel_x;
    let uv = tex_coords[pixel_idx];
    let deriv = derivatives[pixel_idx];

    var color: vec4<f32>;

    switch (params.filter_mode) {
        case 0u: { // Nearest
            let tex_x = u32(apply_boundary(uv.x * f32(params.tex_width), f32(params.tex_width), params.boundary_mode));
            let tex_y = u32(apply_boundary(uv.y * f32(params.tex_height), f32(params.tex_height), params.boundary_mode));
            color = texture_data[tex_y * params.tex_width + tex_x];
        }
        case 1u: { // Bilinear
            color = sample_mip_bilinear(uv, 0u, params.tex_width, params.tex_height, 0u);
        }
        case 2u: { // Trilinear (with mip-mapping)
            let mip_float = compute_mip_level(deriv, f32(params.tex_width), f32(params.tex_height));
            let mip_clamped = clamp(mip_float, 0.0, f32(params.num_mips - 1u));

            let mip0 = u32(floor(mip_clamped));
            let mip1 = min(mip0 + 1u, params.num_mips - 1u);
            let mip_frac = fract(mip_clamped);

            let mip0_width = params.tex_width >> mip0;
            let mip0_height = params.tex_height >> mip0;
            let mip1_width = params.tex_width >> mip1;
            let mip1_height = params.tex_height >> mip1;

            let c0 = sample_mip_bilinear(uv, mip0, mip0_width, mip0_height, mip_offsets[mip0]);
            let c1 = sample_mip_bilinear(uv, mip1, mip1_width, mip1_height, mip_offsets[mip1]);

            color = mix(c0, c1, mip_frac);
        }
        default: {
            color = vec4<f32>(0.0);
        }
    }

    output_colors[pixel_idx] = color;
}

// Backward pass: gradient with respect to texture data
@group(0) @binding(0) var<storage, read> tex_coords_grad: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> grad_output_color: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> grad_texture: array<atomic<u32>>; // Atomic for accumulation
@group(0) @binding(3) var<uniform> grad_tex_params: TextureUniforms;

// Atomically add vec4 (component-wise)
fn atomic_add_vec4(base_idx: u32, value: vec4<f32>) {
    // Store as 4 separate atomic operations
    // In practice, you'd use a proper atomic accumulation strategy
    for (var c = 0u; c < 4u; c++) {
        let idx = base_idx * 4u + c;
        var expected = atomicLoad(&grad_texture[idx]);
        loop {
            let current_f32 = bitcast<f32>(expected);
            let new_f32 = current_f32 + value[c];
            let new_u32 = bitcast<u32>(new_f32);
            let result = atomicCompareExchangeWeak(&grad_texture[idx], expected, new_u32);
            if (result.exchanged) { break; }
            expected = result.old_value;
        }
    }
}

@compute @workgroup_size(16, 16)
fn texture_backward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    if (pixel_x >= grad_tex_params.width || pixel_y >= grad_tex_params.height) {
        return;
    }

    let pixel_idx = pixel_y * grad_tex_params.width + pixel_x;
    let uv = tex_coords_grad[pixel_idx];
    let grad = grad_output_color[pixel_idx];

    // Bilinear backward pass
    let tex_x = uv.x * f32(grad_tex_params.tex_width) - 0.5;
    let tex_y = uv.y * f32(grad_tex_params.tex_height) - 0.5;

    let x0 = u32(floor(apply_boundary(tex_x, f32(grad_tex_params.tex_width), grad_tex_params.boundary_mode)));
    let y0 = u32(floor(apply_boundary(tex_y, f32(grad_tex_params.tex_height), grad_tex_params.boundary_mode)));
    let x1 = u32(floor(apply_boundary(tex_x + 1.0, f32(grad_tex_params.tex_width), grad_tex_params.boundary_mode)));
    let y1 = u32(floor(apply_boundary(tex_y + 1.0, f32(grad_tex_params.tex_height), grad_tex_params.boundary_mode)));

    let fx = fract(tex_x);
    let fy = fract(tex_y);

    // Distribute gradient to the 4 texels based on bilinear weights
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let idx00 = y0 * grad_tex_params.tex_width + x0;
    let idx10 = y0 * grad_tex_params.tex_width + x1;
    let idx01 = y1 * grad_tex_params.tex_width + x0;
    let idx11 = y1 * grad_tex_params.tex_width + x1;

    atomic_add_vec4(idx00, grad * w00);
    atomic_add_vec4(idx10, grad * w10);
    atomic_add_vec4(idx01, grad * w01);
    atomic_add_vec4(idx11, grad * w11);
}
