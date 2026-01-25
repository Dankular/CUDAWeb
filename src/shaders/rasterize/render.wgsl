// Gaussian Splatting Render Shaders
// Vertex and fragment shaders for hardware-accelerated Gaussian rendering
// Uses instanced rendering with per-Gaussian data

struct VertexInput {
    @location(0) position: vec2<f32>,  // Quad corner position (-1 to 1)
};

struct InstanceInput {
    @location(1) center: vec2<f32>,     // Screen-space center
    @location(2) depth: f32,            // Depth for z-buffer
    @location(3) radius: f32,           // Bounding radius
    @location(4) conic_a: f32,          // Conic coefficient a
    @location(5) conic_b: f32,          // Conic coefficient b
    @location(6) conic_c: f32,          // Conic coefficient c
    @location(7) opacity: f32,          // Opacity
    @location(8) color: vec4<f32>,      // RGBA color
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,         // Offset from center
    @location(1) color: vec4<f32>,
    @location(2) conic: vec3<f32>,
    @location(3) opacity: f32,
};

struct RenderUniforms {
    width: f32,
    height: f32,
    near: f32,
    far: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    var output: VertexOutput;

    // Compute screen position
    let screen_pos = instance.center + vertex.position * instance.radius;

    // Convert to NDC
    let ndc_x = (screen_pos.x / uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / uniforms.height) * 2.0;

    // Linearize depth for z-buffer
    let z = (instance.depth - uniforms.near) / (uniforms.far - uniforms.near);

    output.position = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    output.uv = vertex.position * instance.radius;  // Offset in pixels
    output.color = instance.color;
    output.conic = vec3<f32>(instance.conic_a, instance.conic_b, instance.conic_c);
    output.opacity = instance.opacity;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let d = input.uv;
    let conic = input.conic;

    // Evaluate Gaussian: exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
    let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);

    // Discard if too far
    if (power < -4.0) {
        discard;
    }

    let gaussian = exp(power);
    let alpha = input.opacity * gaussian;

    // Discard nearly transparent pixels
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(input.color.rgb, alpha);
}

// Alternative: Full-screen quad with compute-generated texture
struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    var output: FullscreenVertexOutput;

    // Generate fullscreen triangle (more efficient than quad)
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);

    return output;
}

@group(0) @binding(0) var render_texture: texture_2d<f32>;
@group(0) @binding(1) var render_sampler: sampler;

@fragment
fn fs_fullscreen(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(render_texture, render_sampler, input.uv);
}

// Depth-based compositing shader (Order-Independent Transparency)
struct OITOutput {
    @location(0) accumulation: vec4<f32>,  // Pre-multiplied color accumulation
    @location(1) revealage: f32,           // Product of (1 - alpha)
};

@fragment
fn fs_oit(input: VertexOutput) -> OITOutput {
    let d = input.uv;
    let conic = input.conic;

    let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);

    if (power < -4.0) {
        discard;
    }

    let gaussian = exp(power);
    let alpha = input.opacity * gaussian;

    if (alpha < 1.0 / 255.0) {
        discard;
    }

    // Weight function for OIT (McGuire et al. 2013)
    let z = input.position.z;
    let weight = alpha * max(0.01, min(3000.0, 10.0 / (0.00001 + pow(z / 200.0, 4.0))));

    var output: OITOutput;
    output.accumulation = vec4<f32>(input.color.rgb * alpha * weight, alpha * weight);
    output.revealage = 1.0 - alpha;

    return output;
}

// OIT resolve shader
struct OITResolveInput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var accumulation_texture: texture_2d<f32>;
@group(0) @binding(1) var revealage_texture: texture_2d<f32>;
@group(0) @binding(2) var oit_sampler: sampler;

struct OITResolveUniforms {
    background_color: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(3) var<uniform> oit_uniforms: OITResolveUniforms;

@fragment
fn fs_oit_resolve(input: OITResolveInput) -> @location(0) vec4<f32> {
    let accum = textureSample(accumulation_texture, oit_sampler, input.uv);
    let reveal = textureSample(revealage_texture, oit_sampler, input.uv).r;

    // Avoid division by zero
    if (accum.a < 0.00001) {
        return vec4<f32>(oit_uniforms.background_color, 1.0);
    }

    let avg_color = accum.rgb / max(accum.a, 0.00001);
    let alpha = 1.0 - reveal;

    return vec4<f32>(avg_color * alpha + oit_uniforms.background_color * (1.0 - alpha), 1.0);
}
