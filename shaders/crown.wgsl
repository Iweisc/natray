// The crown: the sheet of water the cavity lip throws off at the entry and exit holes, drawn
// over the ray-traced frame like the drops. Each slice of it is the lip material thrown off
// at one moment and carried straight out since, until it tears into the drops that the
// simulation releases at the same moment (lip.wgsl).

// Hole m is shot m / 2, entry or exit.
struct Crown {
    mouth: array<vec4f, 8>,  // xyz centre of the hole, w = cavity source strength there
    axis: array<vec4f, 8>,   // xyz outwards, w = when the bullet passed the hole
    lip: array<vec4f, 8>,    // x bullet silhouette radius, y how long the lip throws off spray, z 1 = on
    misc: vec4f,             // x shown time, y cavity underpressure, z vent time, w balloon radius
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> CR: Crown;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var env_tex: texture_2d<f32>;
@group(0) @binding(4) var env_samp: sampler;

// Around the lip and along the sheet, per hole.
const NPHI: u32 = 384u;
const NROW: u32 = 48u;
const IOR_W: f32 = 1.333;
// Opening per hole in the sheet's surface noise, counted around the lip.
const HOLES: i32 = 64;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f,
    @location(1) n: vec3f,
    @location(2) phi_age: vec2f,
    @location(3) @interpolate(flat) mouth: u32,
}

fn env(d: vec3f, lod: f32) -> vec3f {
    let a = F.sun_col.w;
    let r = vec3f(cos(a) * d.x - sin(a) * d.z, d.y, sin(a) * d.x + cos(a) * d.z);
    let uv = vec2f(atan2(r.z, r.x) / (2.0 * PI) + 0.5, acos(clamp(r.y, -1.0, 1.0)) / PI);
    return textureSampleLevel(env_tex, env_samp, uv, lod).rgb * F.sky.x;
}

fn fresnel(cos_i: f32) -> f32 {
    let eta = 1.0 / IOR_W;
    let sin_t2 = eta * eta * max(0.0, 1.0 - cos_i * cos_i);
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn hash01(x: i32, y: i32) -> f32 {
    return f32(hash3(bitcast<u32>(x), bitcast<u32>(y), 0x9e37u) >> 8u) * (1.0 / 16777216.0);
}

// Value noise that wraps around the lip every `period` cells.
fn vnoise(p: vec2f, period: i32) -> f32 {
    let i = vec2i(floor(p));
    let f = p - vec2f(i);
    let s = f * f * (3.0 - 2.0 * f);
    let x0 = (i.x % period + period) % period;
    let x1 = (x0 + 1) % period;
    return mix(mix(hash01(x0, i.y), hash01(x1, i.y), s.x), mix(hash01(x0, i.y + 1), hash01(x1, i.y + 1), s.x), s.y);
}

fn lip_state(m: u32, tau: f32) -> vec2f {
    return cavity(CR.mouth[m].w, tau, CR.lip[m].x, CR.misc.y, CR.misc.z);
}

fn slice_speed(m: u32, tau: f32) -> f32 {
    return length(lip_velocity(lip_state(m, tau).y));
}

// Distance from the hole's axis and height above the hole of the slice thrown off at
// `tau`, `age` later.
fn profile(m: u32, tau: f32, age: f32) -> vec2f {
    let c = lip_state(m, tau);
    let v = lip_velocity(c.y);
    return vec2f(c.x + v.x * age, -lip_sag(c.x, CR.misc.w) + v.y * age);
}

// Whether even the longest finger of the slice thrown off at `tau` has torn by now.
fn torn(m: u32, t: f32, tau: f32) -> bool {
    return t - tau > 1.6 * sheet_life(slice_speed(m, tau));
}

// Ages of the newest and oldest slice still in one piece somewhere, or an empty range.
fn age_range(m: u32) -> vec2f {
    let t = CR.misc.x - CR.axis[m].w;
    let tau_hi = min(t, CR.lip[m].y);
    if (CR.lip[m].z == 0.0 || tau_hi <= 0.0 || torn(m, t, tau_hi)) { return vec2f(1.0, 0.0); }
    // Later slices are slower and last longer, so what is torn is everything before a cut.
    var lo = 0.0;
    var hi = tau_hi;
    if (torn(m, t, 0.0)) {
        for (var k = 0; k < 16; k++) {
            let mid = 0.5 * (lo + hi);
            if (torn(m, t, mid)) { lo = mid; } else { hi = mid; }
        }
    }
    return vec2f(t - tau_hi, t - lo);
}

fn culled() -> VOut {
    var o: VOut;
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let per = NPHI * NROW * 6u;
    let m = vi / per;
    let q = (vi % per) / 6u;
    let c = vi % 6u;
    let cx = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
    let cy = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let ar = age_range(m);
    if (ar.x >= ar.y) { return culled(); }
    let phi = f32(q % NPHI + cx[c]) / f32(NPHI) * 2.0 * PI;
    let age = mix(ar.x, ar.y, f32(q / NPHI + cy[c]) / f32(NROW));
    let tau = CR.misc.x - CR.axis[m].w - age;
    let pr = profile(m, tau, age);
    let da = 1e-3 * (ar.y - ar.x) + 1e-8;
    let tng = profile(m, tau - da, age + da) - pr;
    let axis = CR.axis[m].xyz;
    // Same basis as the emitter, so the fingers line up with the drops.
    let e = ortho_basis(axis) * vec3f(cos(phi), sin(phi), 0.0);
    let world = CR.mouth[m].xyz + e * pr.x + axis * pr.y;
    let d = world - F.cam_pos.xyz;
    let z = dot(d, F.cam_fwd.xyz);
    let th = F.cam_pos.w;
    var o: VOut;
    o.pos = vec4f(
        (dot(d, F.cam_right.xyz) - dot(F.lens.xyz, F.cam_right.xyz) * z) / (th * F.cam_fwd.w),
        (dot(d, F.cam_up.xyz) - dot(F.lens.xyz, F.cam_up.xyz) * z) / th,
        z - 0.01,
        z);
    o.world = world;
    o.n = normalize(e * tng.y - axis * tng.x);
    o.phi_age = vec2f(phi, age);
    o.mouth = m;
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let m = in.mouth;
    let phi = in.phi_age.x;
    let age = in.phi_age.y;
    let tau = CR.misc.x - CR.axis[m].w - age;
    let u = age / (sheet_life(slice_speed(m, tau)) * (1.0 + 0.6 * finger(phi, m)));
    if (u > 1.0) { discard; }
    // Before a slice tears, holes open in it and join up.
    let lacing = vnoise(vec2f(phi / (2.0 * PI) * f32(HOLES), u * 5.0 + f32(m) * 13.0), HOLES);
    if (lacing < 0.9 * smoothstep(0.45, 1.0, u)) { discard; }
    let d = in.world - F.cam_pos.xyz;
    if (dot(d, F.cam_fwd.xyz) > textureLoad(depth_tex, vec2i(in.pos.xy), 0).r) { discard; }

    let rd = normalize(d);
    let axis = CR.axis[m].xyz;
    let around = ortho_basis(axis) * vec3f(-sin(phi), cos(phi), 0.0);
    // The sheet ripples into ridges running outwards, which grow into the fingers.
    let ridge = vnoise(vec2f(phi / (2.0 * PI) * f32(2 * HOLES), u * 1.5 + f32(m) * 7.0), 2 * HOLES) - 0.5;
    var n = normalize(normalize(in.n) + around * ridge * mix(0.3, 1.2, u));
    if (dot(n, rd) > 0.0) { n = -n; }
    let cosv = clamp(-dot(rd, n), 0.0, 1.0);
    let r1 = fresnel(cosv);
    // Front and back face. The sheet is too thin for the light between them to walk off.
    let rf = 2.0 * r1 / (1.0 + r1);
    var col = env(reflect(rd, n), 0.5) * rf;
    let L = F.sun_dir.xyz;
    let h = normalize(L - rd);
    let a2 = 0.04 * 0.04;
    let nh = abs(dot(n, h));
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    col += F.sun_col.rgb * rf * a2 / (PI * dd * dd) * 0.5;
    // Towards the tearing edge the sheet gathers into a rim like a glass rod, which bends
    // the light behind it away and throws back some of the sky.
    let rim = smoothstep(0.75, 1.0, u);
    let alpha = mix(rf, 0.55, rim);
    col += rim * 0.45 * env(refract(rd, n, 1.0 / IOR_W), 2.0);
    return vec4f(col, alpha);
}
