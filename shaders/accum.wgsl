struct Post {
    rect: vec4f,   // viewport x, y, w, h (pixels)
    res: vec4f,    // render width, height, exposure multiplier, bloom strength
    misc: vec4f,   // x grain seed, y vignette, z accumulated frames, w unused
}

@group(0) @binding(0) var<uniform> PP: Post;
@group(0) @binding(1) var frame_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3u) {
    let w = u32(PP.res.x);
    if (gid.x >= w || gid.y >= u32(PP.res.y)) { return; }
    let c = textureLoad(frame_tex, vec2i(gid.xy), 0);
    let i = gid.y * w + gid.x;
    let n = PP.misc.z;
    accum[i] = select(mix(accum[i], c, 1.0 / (n + 1.0)), c, n < 0.5);
}
