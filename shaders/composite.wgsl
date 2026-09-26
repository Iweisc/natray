struct Post {
    rect: vec4f,
    res: vec4f,
    misc: vec4f,
}

@group(0) @binding(0) var<uniform> PP: Post;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var b0: texture_2d<f32>;
@group(0) @binding(4) var b1: texture_2d<f32>;
@group(0) @binding(5) var b2: texture_2d<f32>;
@group(0) @binding(6) var b3: texture_2d<f32>;
@group(0) @binding(7) var b4: texture_2d<f32>;
@group(0) @binding(8) var b5: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    return VOut(vec4f(p, 0.0, 1.0));
}

fn acc(p: vec2i) -> vec3f {
    let w = i32(PP.res.x);
    let c = clamp(p, vec2i(0), vec2i(w - 1, i32(PP.res.y) - 1));
    return accum[u32(c.y * w + c.x)].rgb;
}

fn agx_contrast(x: vec3f) -> vec3f {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3f) -> vec3f {
    let m = mat3x3f(
        vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let im = mat3x3f(
        vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = m * max(c, vec3f(1e-10));
    v = clamp(log2(v), vec3f(min_ev), vec3f(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_contrast(v);
    // Slightly punchier than base AgX, closer to a camera's default picture profile.
    let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
    v = luma + 1.12 * (v - luma);
    v = pow(max(v, vec3f(0.0)), vec3f(1.08));
    return clamp(im * v, vec3f(0.0), vec3f(1.0));
}

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let uv = (in.pos.xy - PP.rect.xy) / PP.rect.zw;
    let g = uv * PP.res.xy - 0.5;
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    let hdr = mix(mix(acc(i0), acc(i0 + vec2i(1, 0)), f.x), mix(acc(i0 + vec2i(0, 1)), acc(i0 + vec2i(1, 1)), f.x), f.y);
    let bl = (textureSampleLevel(b0, samp, uv, 0.0).rgb + textureSampleLevel(b1, samp, uv, 0.0).rgb
        + textureSampleLevel(b2, samp, uv, 0.0).rgb + textureSampleLevel(b3, samp, uv, 0.0).rgb
        + textureSampleLevel(b4, samp, uv, 0.0).rgb + textureSampleLevel(b5, samp, uv, 0.0).rgb) / 6.0;
    var c = mix(hdr, bl, PP.res.w) * PP.res.z;
    let d = uv - 0.5;
    c *= 1.0 - PP.misc.y * dot(d, d) * 1.6;
    var o = agx(c);
    let n = hash(in.pos.xy + PP.misc.x) + hash(in.pos.xy * 1.37 + PP.misc.x * 0.7) - 1.0;
    o += n * 0.012;
    return vec4f(o, 1.0);
}
