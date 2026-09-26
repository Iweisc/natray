struct Post {
    rect: vec4f,
    res: vec4f,
    misc: vec4f,
}

@group(0) @binding(0) var<uniform> PP: Post;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

fn acc_load(p: vec2i) -> vec3f {
    let w = i32(PP.res.x);
    let c = clamp(p, vec2i(0), vec2i(w - 1, i32(PP.res.y) - 1));
    return accum[u32(c.y * w + c.x)].rgb;
}

// First level reads the full-resolution accumulation buffer (4x4 box around the 2x2 block).
@compute @workgroup_size(8, 8)
fn down_first(@builtin(global_invocation_id) gid: vec3u) {
    let sz = textureDimensions(dst);
    if (gid.x >= sz.x || gid.y >= sz.y) { return; }
    let b = vec2i(gid.xy) * 2;
    var s = vec3f(0.0);
    for (var j = -1; j <= 2; j++) {
        for (var i = -1; i <= 2; i++) {
            let wx = select(1.0, 0.5, i == -1 || i == 2);
            let wy = select(1.0, 0.5, j == -1 || j == 2);
            s += acc_load(b + vec2i(i, j)) * wx * wy;
        }
    }
    textureStore(dst, gid.xy, vec4f(min(s / 9.0, vec3f(20000.0)), 1.0));
}

// 13-tap downsample (Jimenez 2014).
@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3u) {
    let sz = textureDimensions(dst);
    if (gid.x >= sz.x || gid.y >= sz.y) { return; }
    let ss = vec2f(textureDimensions(src));
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(sz);
    let t = 1.0 / ss;
    let a = textureSampleLevel(src, samp, uv + t * vec2f(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(src, samp, uv + t * vec2f(0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(src, samp, uv + t * vec2f(2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(src, samp, uv + t * vec2f(-2.0, 0.0), 0.0).rgb;
    let e = textureSampleLevel(src, samp, uv, 0.0).rgb;
    let f = textureSampleLevel(src, samp, uv + t * vec2f(2.0, 0.0), 0.0).rgb;
    let g = textureSampleLevel(src, samp, uv + t * vec2f(-2.0, 2.0), 0.0).rgb;
    let h = textureSampleLevel(src, samp, uv + t * vec2f(0.0, 2.0), 0.0).rgb;
    let i = textureSampleLevel(src, samp, uv + t * vec2f(2.0, 2.0), 0.0).rgb;
    let j = textureSampleLevel(src, samp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(src, samp, uv + t * vec2f(1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(src, samp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb;
    let m = textureSampleLevel(src, samp, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
    var o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    textureStore(dst, gid.xy, vec4f(o, 1.0));
}
