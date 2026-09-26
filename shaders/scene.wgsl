// Ray tracer: primary rays (trace) and sun photons for ground caustics (photons).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var dens_tex: texture_3d<f32>;
@group(0) @binding(2) var lin_samp: sampler;
@group(0) @binding(3) var<storage, read> W: WGrid;
@group(0) @binding(4) var<storage, read> vindex: array<u32>;
@group(0) @binding(5) var<storage, read> mpos: array<vec4f>;
@group(0) @binding(6) var<storage, read> mnorm: array<vec4f>;
@group(0) @binding(7) var<storage, read> tris: array<Tri>;
@group(0) @binding(8) var<storage, read> alive: array<u32>;
@group(0) @binding(9) var<storage, read> MG: UGrid;
@group(0) @binding(10) var mem_as: acceleration_structure;
@group(0) @binding(11) var env_tex: texture_2d<f32>;
@group(0) @binding(12) var env_samp: sampler;
@group(0) @binding(13) var ground_a: texture_2d<f32>;
@group(0) @binding(14) var ground_b: texture_2d<f32>;
@group(0) @binding(15) var ground_samp: sampler;
@group(0) @binding(16) var<storage, read> wet: array<u32>;
@group(0) @binding(17) var caustic_tex: texture_2d<f32>;
@group(0) @binding(18) var<storage, read> CM: CMap;
@group(0) @binding(19) var mist_tex: texture_3d<f32>;
@group(0) @binding(20) var<storage, read> SG: UGrid;
@group(0) @binding(21) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(22) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(23) var<storage, read_write> caustic_acc: array<atomic<u32>>;
@group(0) @binding(24) var<storage, read> brick_range: array<u32>;

const K_NONE: u32 = 0u;
const K_GROUND: u32 = 1u;
const K_WATER: u32 = 2u;
const K_MEM: u32 = 3u;
const K_BULLET: u32 = 4u;
const K_STRING: u32 = 5u;
const K_KNOT: u32 = 6u;
const IOR_W: f32 = 1.333;
// Absorption of pure water (1/m) at roughly 650, 550 and 450 nm.
const SIGMA_WATER: vec3f = vec3f(0.35, 0.06, 0.018);
const CAUSTIC_RES: u32 = 256u;
const CAUSTIC_SCALE: f32 = 256.0;
const BIG: f32 = 1e20;

struct Hit {
    t: f32,
    kind: u32,
    n: vec3f,
    id: u32,
}

// ---------------------------------------------------------------- environment

fn env_rot(d: vec3f) -> vec3f {
    let a = F.sun_col.w;
    let c = cos(a);
    let s = sin(a);
    return vec3f(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

fn env(d: vec3f, lod: f32) -> vec3f {
    let r = env_rot(d);
    let uv = vec2f(atan2(r.z, r.x) / (2.0 * PI) + 0.5, acos(clamp(r.y, -1.0, 1.0)) / PI);
    return textureSampleLevel(env_tex, env_samp, uv, lod).rgb * F.sky.x;
}

fn sh_irradiance(n0: vec3f) -> vec3f {
    let n = env_rot(n0);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var e = F.sh[0].rgb * 0.282095;
    e += F.sh[1].rgb * 0.488603 * y + F.sh[2].rgb * 0.488603 * z + F.sh[3].rgb * 0.488603 * x;
    e += F.sh[4].rgb * 1.092548 * x * y + F.sh[5].rgb * 1.092548 * y * z;
    e += F.sh[6].rgb * 0.315392 * (3.0 * z * z - 1.0);
    e += F.sh[7].rgb * 1.092548 * x * z + F.sh[8].rgb * 0.546274 * (x * x - y * y);
    return max(e, vec3f(0.0)) * F.sky.x;
}

// Glow of the sun seen through bent paths (refraction smears the disc over a few degrees).
fn sun_lobe(d: vec3f, width: f32) -> vec3f {
    let c = dot(d, F.sun_dir.xyz);
    let a2 = width * width;
    let x = max(1.0 - c, 0.0) * 2.0;
    return F.sun_col.rgb * exp(-x / a2) / (PI * a2);
}

// ---------------------------------------------------------------- BRDF helpers

fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    // eta = n_incident / n_transmitted
    let sin_t2 = eta * eta * max(0.0, 1.0 - cos_i * cos_i);
    if (sin_t2 >= 1.0) { return 1.0; }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn schlick(f0: vec3f, c: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
}

fn ggx_spec(n: vec3f, v: vec3f, l: vec3f, rough: f32, f0: vec3f) -> vec3f {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let a = max(rough * rough, 1e-3);
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let D = a2 / (PI * dd * dd);
    let k = a * 0.5;
    let G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
    return D * G * schlick(f0, dot(v, h)) / (4.0 * nv) ;
}

fn env_brdf_approx(f0: vec3f, rough: f32, nv: f32) -> vec3f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// ---------------------------------------------------------------- water

// Whether a whole brick is on one side of the surface: all below the iso level when
// looking from outside, all above it from inside. Filtering never leaves a brick's texels,
// so a ray can skip such a brick entirely.
fn brick_one_sided(s: u32, inside: bool) -> bool {
    if (s == 0u || s == BRICK_MARKED) { return !inside; }
    if (inside) { return bitcast<f32>(brick_range[2u * s - 1u]) > F.water.x; }
    return bitcast<f32>(brick_range[2u * s - 2u]) < F.water.x;
}

fn slab(ro: vec3f, inv: vec3f, lo: vec3f, hi: vec3f) -> vec2f {
    let t0 = (lo - ro) * inv;
    let t1 = (hi - ro) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

fn safe_inv(d: vec3f) -> vec3f {
    return 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
}

// `lod` grows with the number of surfaces already crossed: deep inside the refraction
// tree a ray contributes little detail, so it may march coarsely.
fn march_water(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, inside: bool, lod: f32) -> f32 {
    if (W.dims.w == 0u) { return -1.0; }
    let inv = safe_inv(rd);
    let margin = W.cell.w * 4.0 + 2.0 * W.cell.x;
    let tb = slab(ro, inv, W.lo.xyz - margin, W.hi.xyz + margin);
    var t = max(tb.x, tmin);
    let tend = min(tb.y, tmax);
    if (t >= tend) { return -1.0; }
    let hstep = (0.9 + 3.0 * lod + select(0.0, 1.2, inside)) * W.cell.x;
    let iso = F.water.x;
    var prev = dens(ro + rd * t) - iso;
    var armed = select((prev < 0.0), (prev > 0.0), inside);
    let bsize = vec3f(8.0 * W.cell.x);
    for (var i = 0; i < 2000; i++) {
        if (t >= tend) { break; }
        if (armed) {
            let p = ro + rd * t;
            let b = floor((p - W.origin.xyz) / bsize);
            if (brick_one_sided(water_slot(vec3i(b)), inside)) {
                let bl = W.origin.xyz + b * bsize;
                let e = slab(ro, inv, bl, bl + bsize);
                t = max(e.y, t) + 0.05 * hstep;
                prev = dens(ro + rd * t) - iso;
                // The surface sits right at the brick face.
                if ((prev >= 0.0) != inside) { return t; }
                continue;
            }
        }
        let tn = min(t + hstep, tend);
        let cur = dens(ro + rd * tn) - iso;
        if (armed) {
            if ((cur >= 0.0) != inside) {
                var a = t;
                var b = tn;
                var fa = prev;
                for (var k = 0; k < select(7, 4, lod > 0.0); k++) {
                    let m = 0.5 * (a + b);
                    let fm = dens(ro + rd * m) - iso;
                    if ((fm >= 0.0) != (fa >= 0.0)) { b = m; } else { a = m; fa = fm; }
                }
                return 0.5 * (a + b);
            }
        } else {
            armed = select((cur < 0.0), (cur > 0.0), inside);
        }
        t = tn;
        prev = cur;
    }
    return -1.0;
}

fn dens_grad(p: vec3f, e: f32) -> vec3f {
    return vec3f(
        dens(p + vec3f(e, 0.0, 0.0)) - dens(p - vec3f(e, 0.0, 0.0)),
        dens(p + vec3f(0.0, e, 0.0)) - dens(p - vec3f(0.0, e, 0.0)),
        dens(p + vec3f(0.0, 0.0, e)) - dens(p - vec3f(0.0, 0.0, e))) / (2.0 * e);
}

fn water_normal(p: vec3f, lod: f32) -> vec3f {
    // The particles leave dents a few spacings wide that read as hammered glass. The wide
    // stencil averages them out, but it reaches across films thinner than itself, so it
    // only takes over where it agrees with the fine one. Deeper in the refraction tree the
    // dents are no longer resolved, so one wide stencil is enough there.
    let gf = dens_grad(p, (1.0 + 2.0 * lod) * W.cell.x);
    var g = gf;
    if (lod == 0.0) {
        let gc = dens_grad(p, 3.0 * W.cell.x);
        g = select(gf, gc, dot(gc, gf) > 0.7 * length(gc) * length(gf));
    }
    let l = length(g);
    return select(vec3f(0.0, 1.0, 0.0), -g / l, l > 1e-8);
}

// ---------------------------------------------------------------- membrane

struct MemHit { t: f32, tri: u32, n: vec3f }

fn trace_mem(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32) -> MemHit {
    var best = MemHit(tmax, 0xffffffffu, vec3f(0.0));
    if (MG.dims.w == 0u || tmax <= tmin) { return best; }
    var rq: ray_query;
    rayQueryInitialize(&rq, mem_as, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, tmin, tmax, ro, rd));
    while (rayQueryProceed(&rq)) {}
    let x = rayQueryGetCommittedIntersection(&rq);
    if (x.kind != RAY_QUERY_INTERSECTION_TRIANGLE) { return best; }
    let t = x.primitive_index;
    let tr = tris[t];
    let u = x.barycentrics.x;
    let v = x.barycentrics.y;
    let n = mnorm[tr.v0].xyz * (1.0 - u - v) + mnorm[tr.v1].xyz * u + mnorm[tr.v2].xyz * v;
    let fn_ = cross(mpos[tr.v1].xyz - mpos[tr.v0].xyz, mpos[tr.v2].xyz - mpos[tr.v0].xyz);
    best.t = x.t;
    best.tri = t;
    best.n = select(normalize(fn_), normalize(n), dot(n, n) > 1e-12);
    return best;
}

fn film_thickness_mm(t: u32) -> f32 {
    let tr = tris[t];
    let p0 = mpos[tr.v0].xyz;
    let a = 0.5 * length(cross(mpos[tr.v1].xyz - p0, mpos[tr.v2].xyz - p0));
    // Rubber is incompressible: thickness scales with rest area / current area.
    return F.rubber.w * clamp(tr.rest_area / max(a, 1e-12), 0.05, 6.0);
}

struct Film { spec: f32, direct: vec3f, scatter: f32, tint: vec3f }

fn film(thick_mm: f32, cosv: f32) -> Film {
    var f: Film;
    f.spec = fresnel_dielectric(cosv, 1.0 / 1.5);
    let path = thick_mm / max(cosv, 0.15);
    let ta = exp(-F.rubber.rgb * path);
    f.scatter = 1.0 - exp(-F.rubber2.w * path);
    f.direct = (1.0 - f.spec) * (1.0 - f.scatter) * ta;
    f.tint = sqrt(ta);
    return f;
}

// ---------------------------------------------------------------- analytic objects

fn bullet_sdf(p: vec3f, bi: u32) -> f32 {
    let B = F.bullets[bi];
    let r = B.pos.w;
    let len = B.dir.w;
    let d = B.dir.xyz;
    let nose = u32(B.look.x);
    if (nose == 3u) { return length(p - (B.pos.xyz - d * r)) - r; }
    let tail = B.pos.xyz - d * len;
    let q = p - tail;
    let u = dot(q, d);
    let qr = q - u * d;
    var rad = length(qr);
    let body = select(select(0.55, 0.42, nose == 1u), 1.0, nose == 2u) * len;
    // Six rifling engravings on the bearing surface. They follow the barrel's twist and turn
    // with the bullet. The depth slope makes this slightly more than a distance, hence the
    // 0.8 on the way out.
    var k = 1.0;
    if (B.look.z > 0.0 && u < body) {
        let e = ortho_basis(d);
        let ang = atan2(dot(qr, e[1]), dot(qr, e[0])) - B.look.y - 2.0 * PI * B.look.z * u;
        let g = abs(fract(ang * 6.0 / (2.0 * PI)) - 0.5) * 2.0;
        rad += 0.03 * r * smoothstep(0.2, 0.14, g);
        k = 0.8;
    }
    if (nose == 2u) {
        // Flat-nosed wadcutter: a cylinder with slightly rounded edges.
        let rr = 0.08 * r;
        let w = vec2f(rad - r + rr, abs(u - 0.5 * len) - 0.5 * len + rr);
        return k * (min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0))) - rr);
    }
    let dc = max(rad - r, max(-u, u - body));
    let x = u - body;
    let nl = len - body;
    var dn: f32;
    if (nose == 1u) {
        // Tangent ogive: in the axial plane it is an arc of radius (r^2 + nl^2) / 2r that
        // meets the cylinder tangentially and ends in a point.
        let ro = (r * r + nl * nl) / (2.0 * r);
        dn = length(vec2f(rad + ro - r, x)) - ro;
    } else {
        // Round nose: half an ellipsoid.
        let ne = vec2f(rad / r, x / nl);
        dn = (length(ne) - 1.0) * min(r, nl);
    }
    return k * min(dc, max(dn, -x));
}

fn bullet_normal(p: vec3f, bi: u32) -> vec3f {
    let e = 1e-5;
    return normalize(vec3f(
        bullet_sdf(p + vec3f(e, 0.0, 0.0), bi) - bullet_sdf(p - vec3f(e, 0.0, 0.0), bi),
        bullet_sdf(p + vec3f(0.0, e, 0.0), bi) - bullet_sdf(p - vec3f(0.0, e, 0.0), bi),
        bullet_sdf(p + vec3f(0.0, 0.0, e), bi) - bullet_sdf(p - vec3f(0.0, 0.0, e), bi)));
}

fn trace_bullet(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, bi: u32) -> f32 {
    let B = F.bullets[bi];
    if (B.pos.w == 0.0) { return -1.0; }
    let len = B.dir.w;
    let c = B.pos.xyz - B.dir.xyz * 0.5 * len;
    let rb = 0.5 * len + B.pos.w;
    let oc = ro - c;
    let b = dot(oc, rd);
    let h = b * b - (dot(oc, oc) - rb * rb);
    if (h < 0.0) { return -1.0; }
    var t = max(-b - sqrt(h), tmin);
    let t1 = min(-b + sqrt(h), tmax);
    for (var i = 0; i < 64; i++) {
        if (t > t1) { break; }
        let d = bullet_sdf(ro + rd * t, bi);
        if (d < 2e-6) { return t; }
        t += d;
    }
    return -1.0;
}

fn trace_string(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32) -> f32 {
    let r = F.wet.w;
    let a = rd.x * rd.x + rd.z * rd.z;
    if (a < 1e-12) { return -1.0; }
    let b = ro.x * rd.x + ro.z * rd.z;
    let c = ro.x * ro.x + ro.z * ro.z - r * r;
    let h = b * b - a * c;
    if (h < 0.0) { return -1.0; }
    let t = (-b - sqrt(h)) / a;
    if (t <= tmin || t >= tmax) { return -1.0; }
    let y = ro.y + rd.y * t;
    if (y < F.knot.y || y > F.wet.z) { return -1.0; }
    return t;
}

fn trace_knot(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32) -> f32 {
    let rad = vec3f(1.0, 0.8, 1.0) * F.knot.w;
    let o = (ro - F.knot.xyz) / rad;
    let d = rd / rad;
    let a = dot(d, d);
    let b = dot(o, d);
    let c = dot(o, o) - 1.0;
    let h = b * b - a * c;
    if (h < 0.0) { return -1.0; }
    let t = (-b - sqrt(h)) / a;
    if (t <= tmin || t >= tmax) { return -1.0; }
    return t;
}

fn trace_objects(ro: vec3f, rd: vec3f, tmin: f32, with_string: bool) -> Hit {
    var h = Hit(BIG, K_NONE, vec3f(0.0), 0u);
    if (rd.y < 0.0 && ro.y > 0.0) {
        let t = -ro.y / rd.y;
        if (t > tmin) { h = Hit(t, K_GROUND, vec3f(0.0, 1.0, 0.0), 0u); }
    }
    for (var bi = 0u; bi < 4u; bi++) {
        let tb = trace_bullet(ro, rd, tmin, h.t, bi);
        if (tb > 0.0) { h = Hit(tb, K_BULLET, bullet_normal(ro + rd * tb, bi), bi); }
    }
    if (with_string) {
        let ts = trace_string(ro, rd, tmin, h.t);
        if (ts > 0.0) {
            let p = ro + rd * ts;
            h = Hit(ts, K_STRING, normalize(vec3f(p.x, 0.0, p.z)), 0u);
        }
    }
    let tk = trace_knot(ro, rd, tmin, h.t);
    if (tk > 0.0) {
        let p = ro + rd * tk;
        let rad = vec3f(1.0, 0.8, 1.0) * F.knot.w;
        h = Hit(tk, K_KNOT, normalize((p - F.knot.xyz) / (rad * rad)), 0u);
    }
    let m = trace_mem(ro, rd, tmin, h.t);
    if (m.tri != 0xffffffffu) { h = Hit(m.t, K_MEM, m.n, m.tri); }
    return h;
}

fn trace_all(ro: vec3f, rd: vec3f, tmin: f32, inside: bool, with_string: bool, lod: f32) -> Hit {
    var h = trace_objects(ro, rd, tmin, with_string);
    let tw = march_water(ro, rd, tmin, h.t, inside, lod);
    if (tw > 0.0) {
        h = Hit(tw, K_WATER, water_normal(ro + rd * tw, lod), 0u);
    }
    return h;
}

// ---------------------------------------------------------------- ground

fn wet_at(p: vec2f) -> f32 {
    let res = u32(F.wet.y);
    let g = (p + F.wet.x) / (2.0 * F.wet.x) * f32(res) - 0.5;
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    var v = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1);
        let c = i0 + o;
        if (any(c < vec2i(0)) || any(c >= vec2i(i32(res)))) { continue; }
        let w = select(1.0 - f, f, vec2<bool>(o.x == 1, o.y == 1));
        v += w.x * w.y * f32(wet[u32(c.y) * res + u32(c.x)]);
    }
    return v;
}

fn caustic_factor(p: vec3f) -> vec3f {
    if (F.flags.x == 0u || CM.ground.w == 0.0) { return vec3f(1.0); }
    let d = p - CM.ground.xyz;
    let u = dot(d, CM.u_axis.xyz) / CM.u_axis.w;
    let v = dot(d, CM.v_axis.xyz) / CM.v_axis.w;
    let m = max(abs(u), abs(v));
    if (m >= 1.0) { return vec3f(1.0); }
    let c = textureSampleLevel(caustic_tex, lin_samp, vec2f(u, v) * 0.5 + 0.5, 0.0).rgb;
    return mix(vec3f(1.0), c, smoothstep(1.0, 0.93, m));
}

// Sun visibility used where photons don't reach (or caustics are disabled).
fn sun_shadow_simple(p: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    if (trace_string(p, L, 1e-4, BIG) > 0.0) { return vec3f(0.08); }
    if (F.flags.x == 0u) {
        let h = trace_objects(p, L, 1e-4, false);
        if (h.kind == K_MEM) {
            let fl = film(film_thickness_mm(h.id), abs(dot(L, h.n)));
            return fl.direct * 0.6;
        }
        if (h.kind == K_KNOT || h.kind == K_BULLET) { return vec3f(0.0); }
        if (march_water(p, L, 1e-4, BIG, false, 1.0) > 0.0) { return vec3f(0.45); }
    }
    return vec3f(1.0);
}

// Sky occlusion from the compact water body (analytic sphere occlusion).
fn ground_sky_occlusion(p: vec3f) -> f32 {
    if (W.dims.w == 0u) { return 1.0; }
    let lo = W.lo.xyz;
    let hi = W.hi.xyz;
    let c = 0.5 * (lo + hi);
    let r = 0.5 * min(hi.x - lo.x, min(hi.y - lo.y, hi.z - lo.z));
    let d = c - p;
    let dl = length(d);
    if (r <= 0.0 || dl < 1e-4) { return 1.0; }
    let s = r / dl;
    return 1.0 - s * s * max(d.y / dl, 0.0) * 0.9;
}

fn ground_shade(p: vec3f, rd: vec3f, footprint: f32, cheap: bool) -> vec3f {
    let tile = F.sky.w;
    let uv = p.xz / tile;
    // Between the footprint's short and long axis at grazing angles. Accumulation takes care of
    // the aliasing that is left along the long axis.
    let lod = log2(max(footprint / sqrt(max(abs(rd.y), 0.05)) / (tile / 2048.0), 1.0));
    var albedo = textureSampleLevel(ground_a, ground_samp, uv, lod).rgb;
    let gb = textureSampleLevel(ground_b, ground_samp, uv, lod);
    let nm = gb.xy * 2.0 - 1.0;
    var n = normalize(vec3f(nm.x, sqrt(max(1.0 - dot(nm, nm), 0.0)), -nm.y));
    var rough = gb.z;
    let ao = gb.w;
    // The wet map holds water volume per cell in 1e-3 mm^3. A cell is 5.5 mm^2, so 3000 is
    // a 0.5 mm film: soaked pores. Past a couple of mm it is a puddle with a flat surface.
    let water = wet_at(p.xz);
    let wet = smoothstep(150.0, 3000.0, water);
    let puddle = smoothstep(4000.0, 25000.0, water);
    // Wet concrete: darker (light gets trapped in the water-filled pores) and glossier.
    albedo *= mix(1.0, 0.45, wet);
    rough = mix(rough, 0.18, wet);
    n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), wet * 0.8));

    let L = F.sun_dir.xyz;
    let v = -rd;
    var vis = sun_shadow_simple(p);
    if (!cheap) { vis *= caustic_factor(p); }
    let sun = F.sun_col.rgb * vis;
    let nl = max(dot(n, L), 0.0);
    let f0 = vec3f(0.04);
    var col = albedo / PI * (sun * nl + sh_irradiance(n) * ao * ground_sky_occlusion(p));
    col += ggx_spec(n, v, L, max(rough, 0.05), f0) * sun * nl;
    let r = reflect(rd, n);
    let nv = max(dot(n, v), 1e-3);
    col += env(r, clamp(rough * 9.0, 0.0, 10.0)) * env_brdf_approx(f0, rough, nv) * ao * mix(0.3, 1.0, wet);
    if (puddle > 0.0) {
        let up = vec3f(0.0, 1.0, 0.0);
        let fw = fresnel_dielectric(max(-rd.y, 0.0), 1.0 / IOR_W);
        let film = env(reflect(rd, up), 0.5) * fw
            + ggx_spec(up, v, L, 0.03, vec3f(0.02)) * sun * max(L.y, 0.0);
        col = mix(col, col * (1.0 - fw) + film, puddle);
    }

    // Far away the textured plane fades into the photographed ground of the HDRI.
    let dist = length(p.xz - F.cam_pos.xz);
    return mix(col, env(rd, F.jitter.w), smoothstep(6.0, 18.0, dist));
}

// ---------------------------------------------------------------- shading

fn sun_visibility(p: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    var vis = vec3f(1.0);
    var ro = p;
    for (var i = 0; i < 6; i++) {
        let h = trace_objects(ro, L, 1e-5, true);
        if (h.kind == K_NONE || h.kind == K_GROUND) { break; }
        if (h.kind != K_MEM) { return vec3f(0.0); }
        let fl = film(film_thickness_mm(h.id), abs(dot(L, h.n)));
        vis *= fl.direct + 0.5 * fl.scatter * F.rubber2.rgb * fl.tint;
        ro = ro + L * (h.t + 1e-5);
        if (max(vis.x, max(vis.y, vis.z)) < 0.01) { return vec3f(0.0); }
    }
    return vis;
}

fn shade_rubber(p: vec3f, n_in: vec3f, rd: vec3f, fl: Film, rough: f32, lod: f32) -> vec3f {
    let n = select(n_in, -n_in, dot(n_in, rd) > 0.0);
    let v = -rd;
    let L = F.sun_dir.xyz;
    let vis = sun_visibility(p);
    let sun = F.sun_col.rgb * vis;
    let alb = F.rubber2.rgb * fl.tint;
    let e_front = sun * max(dot(n, L), 0.0) + sh_irradiance(n);
    let e_back = sun * max(dot(-n, L), 0.0) + sh_irradiance(-n);
    var col = (1.0 - fl.spec) * fl.scatter * 0.5 * alb / PI * (e_front + e_back);
    col += ggx_spec(n, v, L, rough, vec3f(0.04)) * sun * max(dot(n, L), 0.0);
    let r = reflect(rd, n);
    col += env(r, lod) * fl.spec;
    return col;
}

fn shade_bullet(p: vec3f, n: vec3f, rd: vec3f, bi: u32) -> vec3f {
    let metal = u32(F.bullets[bi].look.w);
    // Dull oxidised lead, a copper jacket or polished steel.
    let f0 = select(select(vec3f(0.5, 0.5, 0.53), vec3f(0.955, 0.638, 0.538), metal == 1u), vec3f(0.56, 0.57, 0.58), metal == 2u);
    let v = -rd;
    let L = F.sun_dir.xyz;
    let rough = select(select(0.55, 0.28, metal == 1u), 0.12, metal == 2u);
    let nv = max(dot(n, v), 1e-3);
    var col = env(reflect(rd, n), clamp(rough * 9.0, 0.0, 10.0)) * env_brdf_approx(f0, rough, nv);
    col += ggx_spec(n, v, L, rough, f0) * F.sun_col.rgb * max(dot(n, L), 0.0);
    return col;
}

fn shade_diffuse(p: vec3f, n: vec3f, alb: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    let vis = sun_visibility(p + n * 1e-4);
    return alb / PI * (F.sun_col.rgb * vis * max(dot(n, L), 0.0) + sh_irradiance(n));
}

// Radiance along a secondary (reflected) direction without tracing more water.
fn cheap_radiance(p: vec3f, d: vec3f, lod: f32) -> vec3f {
    if (d.y < 0.0 && p.y > 0.0) {
        let t = -p.y / d.y;
        return ground_shade(p + d * t, d, 0.004 * t, false);
    }
    return env(d, lod);
}

fn mist_march(ro: vec3f, rd: vec3f, tmax: f32, seed: ptr<function, u32>) -> vec4f {
    if (F.flags.z == 0u || SG.dims.w == 0u) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let cs = SG.origin.w;
    let lo = SG.origin.xyz;
    let hi = lo + (vec3f(SG.dims.xyz) - 1.0) * cs;
    let tb = slab(ro, safe_inv(rd), lo, hi);
    let t0 = max(tb.x, 0.0);
    let t1 = min(tb.y, tmax);
    if (t0 >= t1) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    // About a voxel per step. Rays that only clip a corner of the box need far fewer.
    let n = clamp(i32(ceil((t1 - t0) / cs)), 4, 64);
    let ds = (t1 - t0) / f32(n);
    var t = t0 + ds * rnd(seed);
    var T = 1.0;
    var L = vec3f(0.0);
    let cosang = dot(rd, F.sun_dir.xyz);
    // Droplets of 5-25 um: strong forward diffraction lobe plus a broad part.
    let g1 = 0.86;
    let g2 = 0.2;
    let hg1 = (1.0 - g1 * g1) / (4.0 * PI * pow(1.0 + g1 * g1 - 2.0 * g1 * cosang, 1.5));
    let hg2 = (1.0 - g2 * g2) / (4.0 * PI * pow(1.0 + g2 * g2 - 2.0 * g2 * cosang, 1.5));
    let phase = 0.7 * hg1 + 0.3 * hg2;
    let iso = 1.0 / (4.0 * PI);
    let amb = sh_irradiance(vec3f(0.0, 1.0, 0.0)) * 0.6 / (4.0 * PI) + sh_irradiance(vec3f(0.0, -1.0, 0.0)) * 0.4 / (4.0 * PI);
    for (var i = 0; i < n; i++) {
        let p = ro + rd * t;
        let uvw = ((p - lo) / cs + 0.5) / 128.0;
        // x extinction, y/z the sun's octave sums, w the ambient gain (prep.wgsl mist_shadow).
        let m = textureSampleLevel(mist_tex, lin_samp, uvw, 0.0);
        if (m.x > 1e-3) {
            let li = F.sun_col.rgb * (m.y * iso + m.z * (phase - iso)) + amb * m.w;
            let a = exp(-m.x * ds);
            L += T * li * (1.0 - a);
            T *= a;
            if (T < 0.01) { break; }
        }
        t += ds;
    }
    return vec4f(L, T);
}

struct Shaded { col: vec3f, depth: f32 }

fn shade(ro0: vec3f, rd0: vec3f, seed: ptr<function, u32>) -> Shaded {
    var out: Shaded;
    out.depth = BIG;
    var L = vec3f(0.0);
    var T = vec3f(1.0);
    var ro = ro0;
    var rd = rd0;
    var inside = false;
    var seg_len_total = 0.0;
    let lod_bg = F.jitter.w;
    let max_seg = i32(F.flags.y) + 4;
    for (var seg = 0; seg < max_seg; seg++) {
        let h = trace_all(ro, rd, 1e-5, inside, true, f32(max(seg - 1, 0)));
        if (seg == 0) {
            out.depth = h.t;
            let m = mist_march(ro, rd, h.t, seed);
            L += m.rgb;
            T *= m.a;
        }
        let tt = select(h.t, 0.0, h.kind == K_NONE);
        if (inside) { T *= exp(-SIGMA_WATER * tt); }
        seg_len_total += tt;
        let p = ro + rd * h.t;
        let footprint = F.res.w * seg_len_total;
        if (h.kind == K_NONE) {
            L += T * env(rd, lod_bg);
            if (seg > 0) { L += T * sun_lobe(rd, 0.035); }
            break;
        }
        if (h.kind == K_GROUND) {
            L += T * ground_shade(p, rd, footprint, false);
            break;
        }
        if (h.kind == K_BULLET) {
            L += T * shade_bullet(p, h.n, rd, h.id);
            break;
        }
        if (h.kind == K_STRING) {
            L += T * shade_diffuse(p, h.n, vec3f(0.62, 0.6, 0.55));
            break;
        }
        if (h.kind == K_KNOT) {
            // Many layers of rubber: light that comes back out has been filtered repeatedly.
            L += T * (shade_diffuse(p, h.n, F.rubber2.rgb * F.rubber2.rgb)
                + env(reflect(rd, h.n), 2.5) * schlick(vec3f(0.04), dot(-rd, h.n)));
            break;
        }
        if (h.kind == K_MEM) {
            let cosv = abs(dot(rd, h.n));
            let thick = film_thickness_mm(h.id);
            let fl = film(thick, cosv);
            // Stretched latex is glossy. Once it snaps back its surface crinkles to a satin.
            let rough = mix(0.16, 0.35, smoothstep(0.06, 0.2, thick));
            L += T * shade_rubber(p, h.n, rd, fl, rough, rough * 9.0);
            T *= fl.direct;
            ro = p + rd * 2e-5;
            if (max(T.x, max(T.y, T.z)) < 0.004) { break; }
            continue;
        }
        // Water interface.
        let n = h.n;
        let nf = select(n, -n, inside);
        let cos_i = clamp(-dot(rd, nf), 0.0, 1.0);
        let eta = select(1.0 / IOR_W, IOR_W, inside);
        let fr = fresnel_dielectric(cos_i, eta);
        let refl = reflect(rd, nf);
        if (!inside) {
            L += T * fr * cheap_radiance(p, refl, 0.5);
            L += T * ggx_spec(nf, -rd, F.sun_dir.xyz, 0.035, vec3f(0.02)) * F.sun_col.rgb * max(dot(nf, F.sun_dir.xyz), 0.0);
        }
        let refr = refract(rd, nf, eta);
        let step = 0.35 * W.cell.x;
        if (dot(refr, refr) < 1e-6) {
            rd = refl;
        } else {
            if (inside) {
                // Light leaving through the surface from inside also reflects back in; we
                // follow the transmitted path and add the reflected part as environment.
                L += T * fr * env(refl, 1.5);
            }
            T *= 1.0 - fr;
            rd = refr;
        }
        ro = p + rd * step;
        // The smoothed normal can send a ray to the other side of the actual surface.
        // Toggling would then absorb water colour over metres of air.
        inside = dens(ro) > F.water.x;
        if (max(T.x, max(T.y, T.z)) < 0.004) { break; }
        if (seg == max_seg - 1) { L += T * env(rd, lod_bg); }
    }
    out.col = L;
    return out;
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) gid: vec3u) {
    if (f32(gid.x) >= F.res.x || f32(gid.y) >= F.res.y) { return; }
    var seed = hash3(gid.x, gid.y, bitcast<u32>(F.jitter.z));
    let px = vec2f(gid.xy) + 0.5 + F.jitter.xy;
    let ndc = vec2f(px.x / F.res.x * 2.0 - 1.0, 1.0 - px.y / F.res.y * 2.0);
    let th = F.cam_pos.w;
    let rd = normalize(F.cam_fwd.xyz + ndc.x * th * F.cam_fwd.w * F.cam_right.xyz + ndc.y * th * F.cam_up.xyz + F.lens.xyz);
    let s = shade(F.cam_pos.xyz, rd, &seed);
    textureStore(out_color, gid.xy, vec4f(min(s.col, vec3f(60000.0)), 1.0));
    textureStore(out_depth, gid.xy, vec4f(s.depth * dot(rd, F.cam_fwd.xyz), 0.0, 0.0, 0.0));
}

fn deposit(p: vec3f, flux: vec3f) {
    let d = p - CM.ground.xyz;
    let u = dot(d, CM.u_axis.xyz) / CM.u_axis.w;
    let v = dot(d, CM.v_axis.xyz) / CM.v_axis.w;
    let g = (vec2f(u, v) * 0.5 + 0.5) * f32(CAUSTIC_RES) - 0.5;
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1);
        let c = i0 + o;
        if (any(c < vec2i(0)) || any(c >= vec2i(i32(CAUSTIC_RES)))) { continue; }
        let w = select(1.0 - f, f, vec2<bool>(o.x == 1, o.y == 1));
        let idx = (u32(c.y) * CAUSTIC_RES + u32(c.x)) * 3u;
        let q = flux * (w.x * w.y * CAUSTIC_SCALE);
        atomicAdd(&caustic_acc[idx], u32(q.x));
        atomicAdd(&caustic_acc[idx + 1u], u32(q.y));
        atomicAdd(&caustic_acc[idx + 2u], u32(q.z));
    }
}

@compute @workgroup_size(8, 8)
fn photons(@builtin(global_invocation_id) gid: vec3u) {
    let n = F.counts.w;
    if (gid.x >= n || gid.y >= n || CM.ground.w == 0.0 || F.flags.x == 0u) { return; }
    var seed = hash3(gid.x, gid.y, bitcast<u32>(F.jitter.z) ^ 0x7777u);
    let L = F.sun_dir.xyz;
    let e1 = CM.v_axis.xyz;
    let e2 = normalize(cross(L, e1));
    let s1 = (f32(gid.x) + rnd(&seed)) / f32(n) * 2.0 - 1.0;
    let s2 = (f32(gid.y) + rnd(&seed)) / f32(n) * 2.0 - 1.0;
    let r = CM.center.w;
    var ro = CM.center.xyz + (e1 * s1 + e2 * s2) * r + L * (r + 0.5);
    var rd = -L;
    var flux = vec3f(1.0);
    var inside = false;
    for (var seg = 0; seg < 10; seg++) {
        let h = trace_all(ro, rd, 1e-5, inside, false, 1.0);
        if (inside) { flux *= exp(-SIGMA_WATER * select(h.t, 0.0, h.kind == K_NONE)); }
        if (h.kind == K_NONE) { return; }
        let p = ro + rd * h.t;
        if (h.kind == K_GROUND) {
            deposit(p, flux);
            return;
        }
        if (h.kind == K_MEM) {
            let fl = film(film_thickness_mm(h.id), abs(dot(rd, h.n)));
            flux *= fl.direct + 0.5 * fl.scatter * F.rubber2.rgb * fl.tint;
            ro = p + rd * 2e-5;
            continue;
        }
        if (h.kind != K_WATER) { return; }
        let nf = select(h.n, -h.n, inside);
        let cos_i = clamp(-dot(rd, nf), 0.0, 1.0);
        let eta = select(1.0 / IOR_W, IOR_W, inside);
        let refr = refract(rd, nf, eta);
        let step = 0.35 * W.cell.x;
        if (dot(refr, refr) < 1e-6) {
            rd = reflect(rd, nf);
        } else {
            flux *= 1.0 - fresnel_dielectric(cos_i, eta);
            rd = refr;
        }
        ro = p + rd * step;
        inside = dens(ro) > F.water.x;
        if (max(flux.x, max(flux.y, flux.z)) < 0.01) { return; }
    }
}
