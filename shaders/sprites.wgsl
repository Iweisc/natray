// Spray droplets and detached water particles, drawn as refracting sphere impostors
// over the ray-traced frame. Mist particles are rendered volumetrically instead.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> spray: array<Spray>;
@group(0) @binding(2) var<storage, read> P: array<Particle>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var env_tex: texture_2d<f32>;
@group(0) @binding(5) var env_samp: sampler;
@group(0) @binding(6) var dens_tex: texture_3d<f32>;
@group(0) @binding(7) var lin_samp: sampler;
@group(0) @binding(8) var<storage, read> W: WGrid;
@group(0) @binding(9) var<storage, read> spos: array<vec4f>;
@group(0) @binding(10) var<storage, read> vindex: array<u32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) q: vec2f,
    @location(1) @interpolate(flat) info: vec4f,    // radius px, streak px, view depth, alpha scale
    @location(2) @interpolate(flat) center: vec4f,  // world centre, world radius
    @location(3) @interpolate(flat) axes: vec4f,    // streak direction (screen), perpendicular
}

fn env(d: vec3f, lod: f32) -> vec3f {
    let a = F.sun_col.w;
    let r = vec3f(cos(a) * d.x - sin(a) * d.z, d.y, sin(a) * d.x + cos(a) * d.z);
    let uv = vec2f(atan2(r.z, r.x) / (2.0 * PI) + 0.5, acos(clamp(r.y, -1.0, 1.0)) / PI);
    return textureSampleLevel(env_tex, env_samp, uv, lod).rgb * F.sky.x;
}

fn culled() -> VOut {
    var o: VOut;
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var c: vec3f;
    var r: f32;
    var vel: vec3f;
    if (ii < F.counts.y) {
        let s = spray[ii];
        if (s.pos.w <= 0.0 || s.vel.w != 1.0) { return culled(); }
        c = s.pos.xyz;
        r = s.pos.w;
        vel = s.vel.xyz;
    } else {
        let j = ii - F.counts.y;
        if (j >= F.counts.x) { return culled(); }
        let v = P[j].vel;
        if (v.w < 0.5 || W.dims.w == 0u) { return culled(); }
        // Same smoothed position the density field was built from.
        c = spos[j].xyz;
        // Particles that are part of the surface are drawn by the ray tracer.
        if (dens(c) > 0.25) { return culled(); }
        r = 0.55 * F.water.y;
        vel = v.xyz;
    }
    let d = c - F.cam_pos.xyz;
    let z = dot(d, F.cam_fwd.xyz);
    if (z < 0.02) { return culled(); }
    let th = F.cam_pos.w;
    let asp = F.cam_fwd.w;
    let half = vec2f(F.res.x, F.res.y) * 0.5;
    let shift = vec2f(dot(F.lens.xyz, F.cam_right.xyz), dot(F.lens.xyz, F.cam_up.xyz));
    let ndc = vec2f((dot(d, F.cam_right.xyz) / z - shift.x) / (th * asp), (dot(d, F.cam_up.xyz) / z - shift.y) / th);
    let cpx = ndc * half;
    let rad_px = r / (z * th) * half.y;
    let size = max(rad_px, 0.7);
    let d2 = d + vel * F.time.y;
    let z2 = max(dot(d2, F.cam_fwd.xyz), 0.02);
    let cpx2 = vec2f((dot(d2, F.cam_right.xyz) / z2 - shift.x) / (th * asp), (dot(d2, F.cam_up.xyz) / z2 - shift.y) / th) * half;
    let streak = cpx2 - cpx;
    let slen = min(length(streak), 400.0);
    let a = select(vec2f(1.0, 0.0), streak / max(length(streak), 1e-6), slen > 0.01);
    let b = vec2f(-a.y, a.x);
    let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let k = corners[vi];
    let ea = size + 0.5 * slen + 1.0;
    let eb = size + 1.0;
    let mid = cpx + 0.5 * streak;
    let ppx = mid + a * k.x * ea + b * k.y * eb;
    var o: VOut;
    o.pos = vec4f(ppx / half, 0.5, 1.0);
    o.q = vec2f(k.x * ea, k.y * eb);
    let cover = min(1.0, (rad_px / size) * (rad_px / size));
    o.info = vec4f(size, slen, z, cover * (2.0 * size) / (2.0 * size + slen));
    o.center = vec4f(c, r);
    o.axes = vec4f(a, b);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let scene_z = textureLoad(depth_tex, vec2i(in.pos.xy), 0).r;
    if (in.info.z > scene_z) { discard; }
    let size = in.info.x;
    let slen = in.info.y;
    let qa = max(abs(in.q.x) - 0.5 * slen, 0.0) * sign(in.q.x);
    let dq = vec2f(qa, in.q.y);
    let dd = length(dq);
    let cov = clamp(size + 0.5 - dd, 0.0, 1.0);
    if (cov <= 0.0) { discard; }
    let nq = dq / max(size, 1e-3) * min(1.0, 0.98 * size / max(dd, 1e-4));
    let nz = sqrt(max(1.0 - dot(nq, nq), 0.0));
    let sa = in.axes.xy;
    let sb = in.axes.zw;
    let sx = sa * nq.x + sb * nq.y;
    let n = normalize(F.cam_right.xyz * sx.x + F.cam_up.xyz * sx.y - F.cam_fwd.xyz * nz);
    let rd = normalize(in.center.xyz - F.cam_pos.xyz);
    let cosv = clamp(dot(n, -rd), 0.0, 1.0);
    let fr = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
    let rf = reflect(rd, n);
    var col = env(rf, 2.0) * fr;
    // Sun glint: roughness grows as the droplet shrinks below a pixel so energy is not lost.
    let L = F.sun_dir.xyz;
    let h = normalize(L - rd);
    let a = max(0.03, 0.6 / max(size, 0.5));
    let a2 = a * a;
    let nh = max(dot(n, h), 0.0);
    let dd2 = nh * nh * (a2 - 1.0) + 1.0;
    col += F.sun_col.rgb * fr * a2 / (PI * dd2 * dd2) * 0.25 * select(0.0, 1.0, dot(n, L) > 0.0);
    // Ball lens: refract in, cross the sphere, refract out.
    let d1 = refract(rd, n, 1.0 / 1.333);
    let p2 = n + d1 * (-2.0 * dot(n, d1));
    let n2 = normalize(p2);
    var d2 = refract(d1, -n2, 1.333);
    if (dot(d2, d2) < 1e-6) { d2 = reflect(d1, -n2); }
    let c = dot(d2, L);
    let lobe = F.sun_col.rgb * exp(-max(1.0 - c, 0.0) * 2.0 / 0.0025) / (PI * 0.0025);
    col += (1.0 - fr) * (1.0 - fr) * (env(d2, 1.5) + lobe * 0.3);
    let alpha = cov * in.info.w;
    return vec4f(col * alpha, alpha);
}
