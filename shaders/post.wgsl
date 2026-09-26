// Caustic map resolve, progressive accumulation, bloom pyramid and final tone mapping.

struct Post {
    rect: vec4f,   // viewport x, y, w, h (pixels)
    res: vec4f,    // render width, height, exposure multiplier, bloom strength
    misc: vec4f,   // x grain seed, y vignette, z accumulated frames, w unused
}

// ---------------------------------------------------------------- caustics

@group(0) @binding(0) var<uniform> CF: Frame;
@group(0) @binding(1) var<storage, read_write> c_acc: array<u32>;
@group(0) @binding(2) var c_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn caustic_resolve(@builtin(global_invocation_id) gid: vec3u) {
    let res = 256u;
    if (gid.x >= res || gid.y >= res) { return; }
    let n = f32(CF.counts.w);
    let expected = n * n * max(CF.time.z, 1.0) / f32(res * res);
    let k = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
    var s = vec3f(0.0);
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let c = clamp(vec2i(gid.xy) + vec2i(i, j), vec2i(0), vec2i(i32(res) - 1));
            let idx = (u32(c.y) * res + u32(c.x)) * 3u;
            s += k[i + 2] * k[j + 2] * vec3f(f32(c_acc[idx]), f32(c_acc[idx + 1u]), f32(c_acc[idx + 2u]));
        }
    }
    s /= 256.0 * 256.0 * expected;
    textureStore(c_out, gid.xy, vec4f(s, 1.0));
}
