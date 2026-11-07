// Bullets: rigid bodies of revolution, stepped with the water.
//
// The bullet is far thinner than a grid cell, so the grid can't show the flow around it.
// Instead every patch of its surface takes the pressure a patch at that angle takes in
// water: the cavitating-cone pressure where it moves into water, the wake's where it moves
// away, never below the vapour pressure, and the gas's pressure where there is no water. The
// water takes the opposite push.
//
// The water the nose pushes aside moves off sideways far faster than any cell, and that
// motion, not the push along the path, is where the bullet's energy goes (Logvinovich: each
// slice of the cavity takes the drag's work as it opens). Each step the nose leaves a section
// of cavity that grows at the rate the drag sets. While it is thinner than the grid it feeds
// the water's pressure solve as a source of volume, and moves particles out of it. Once it
// is wider, the grid carries on by itself.

struct Bullet {
    // Centre of mass, w = mass.
    x: vec4f,
    // w = roll angle about the axis, for the renderer.
    v: vec4f,
    // Body to world. The body's +z runs from the base to the nose.
    q: vec4f,
    // Angular momentum about the centre of mass, w = the speed the step length has to allow
    // for.
    l: vec4f,
    // x axial moment of inertia, y transverse, z how far the surface gets from the centre of
    // mass, w radius.
    shape: vec4f,
    // Around every section still thinner than the grid, for the particles to check first.
    reach: vec4f,
    // x when it arrives (sim ns), y patch count, z 0 on its way / 1 in the grid / 2 gone,
    // w sections left so far.
    info: vec4u,
}

struct Patch {
    // Body frame, w = area.
    p: vec4f,
    n: vec4f,
}

// A stretch of the path of the point where the flow leaves the wetted face, and the cavity
// it opened. a.w = when (s), b.w = how fast the cavity's cross-section grows (d(R^2)/dt),
// c.x = its radius squared to begin with. A cavity is the cylinder along the stretch: the
// face that makes the next one is just ahead of it.
struct Section {
    a: vec4f,
    b: vec4f,
    c: vec4f,
}

// x.w = t_ns (bits) of the step, 0 when the bullet wasn't in the grid. v.w = roll.
struct PoseEntry {
    x: vec4f,
    q: vec4f,
    v: vec4f,
}

@group(0) @binding(32) var<storage, read_write> bullets: array<Bullet>;
@group(0) @binding(33) var<storage, read_write> patches: array<Patch>;
@group(0) @binding(34) var<storage, read_write> sections: array<Section>;
@group(0) @binding(35) var<storage, read_write> pose_log: array<PoseEntry>;
// Each bullet's outline, OUTLINE points of (z, r) from the base to the nose: z along the
// axis from the centre of mass, r the radius there.
@group(0) @binding(36) var<storage, read_write> outline: array<vec2f>;
// What the rubber has done to each bullet since its last step: impulse, then angular impulse
// about the centre of mass. Then the rubber it held: its outward normal, its place from the
// centre of mass (mm), each times its area (mm^2), and the area.
@group(0) @binding(37) var<storage, read_write> bullet_hit: array<atomic<i32>>;

const MAX_PATCHES: u32 = 576u;
const N_SEC: u32 = 32u;
const LOG_CAP: u32 = 256u;
const OUTLINE: u32 = 33u;
const HIT_FIELDS: u32 = 13u;
// N s and N m s, to about 2 and 0.03 per step.
const FP_J: f32 = 1073741824.0;
const FP_L: f32 = 68719476736.0;
// Area times place, to about 5e5 mm^3.
const FP_X: f32 = 4096.0;
// Dry latex sliding on a copper jacket.
const MU_JACKET: f32 = 0.8;
// ln of how far the cavity's flow reaches over its radius, in Logvinovich's energy balance.
const LOGVINOVICH_K: f32 = 2.0;
// A section this wide is the grid's.
const R_HAND: f32 = 1.5;
// Wake pressure behind a bluff body, as a share of the dynamic pressure.
const CP_WAKE: f32 = -0.4;
// Turbulent skin friction.
const CF_SKIN: f32 = 0.003;

fn qrot(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn qmul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// Pressure coefficient of a patch whose surface meets the flow at an angle with sine `s`,
// positive when it moves into the water. Moving into it, the pressure on a cone of that
// half-angle in a vapour cavity (Guzevsky's fit, from 15 degrees to a flat disk), falling off
// with the square of the sine below that. Moving away, the wake's.
fn cp_patch(s: f32) -> f32 {
    if (s < 0.0) { return CP_WAKE * s * s; }
    let beta = asin(min(s, 1.0));
    if (beta < PI / 12.0) { return 2.13 * s * s; }
    let x = beta / PI - 0.25;
    return 0.5 + 1.81 * x - 2.0 * x * x;
}

// Gauge pressure around `x` (in cells, on the lattice of cell centres) over the cells that
// aren't solid.
fn sample_pressure(x: vec3f, gas_only: bool) -> vec2f {
    let base = vec3i(floor(x - 0.5));
    let fx = x - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    var s = vec2f(0.0);
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let cell = cell_at(base + vec3i(vec3u(a, b, c)));
                if (cell == NONE) { continue; }
                let ty = cf[cf_i(F_TYPE, cell)];
                if (ty == SOLID || (gas_only && ty == WATER)) { continue; }
                let w = wx[a] * wy[b] * wz[c];
                s += vec2f(w * cf[cf_i(F_P, cell)], w);
            }
        }
    }
    return s;
}

// Hands `j` (N s) along `d` to the water faces around `x` (in cells), in proportion to how
// much each weighs in the sum `sw` of weight times water there.
fn push_water(x: vec3f, d: u32, j: f32, sw: f32) {
    if (sw <= 1e-6 || j == 0.0) { return; }
    let lx = x - lattice_offset(d);
    let base = vec3i(floor(lx - 0.5));
    let fx = lx - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    let hs = U.fluid.w;
    let k = j / (U.fluid.x * hs * hs * hs * sw);
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let cell = cell_at(base + vec3i(vec3u(a, b, c)));
                if (cell == NONE || atomicLoad(&acc[acc_i(a_wt(d), cell)]) <= 0) { continue; }
                atomicAdd(&acc[acc_i(A_BU + d, cell)], to_fp(k * wx[a] * wy[b] * wz[c], FP_M));
            }
        }
    }
}

fn section_r2(s: Section, t: f32) -> f32 {
    return s.c.x + s.b.w * max(t - s.a.w, 0.0);
}

// Whether `p` is inside one of the bullet's cavity sections still thinner than the grid.
fn in_cavity(b: u32, p: vec3f, t: f32) -> bool {
    let hand2 = R_HAND * R_HAND * h() * h();
    for (var k = 0u; k < N_SEC; k++) {
        let s = sections[b * N_SEC + k];
        if (s.b.w <= 0.0 || section_r2(s, t) >= hand2) { continue; }
        let ab = s.b.xyz - s.a.xyz;
        let f = dot(p - s.a.xyz, ab) / max(dot(ab, ab), 1e-20);
        if (f < 0.0 || f > 1.0) { continue; }
        let d = p - (s.a.xyz + f * ab);
        if (dot(d, d) < section_r2(s, t)) { return true; }
    }
    return false;
}

var<workgroup> red_f: array<vec4f, 256>;
var<workgroup> red_t: array<vec4f, 256>;
var<workgroup> red_c: array<vec4f, 256>;
var<workgroup> p_far: f32;
var<workgroup> flying: u32;
var<workgroup> held_side: vec4f;

@compute @workgroup_size(256)
fn bullet_step(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let bi = wid.x;
    let b = bullets[bi];
    let step = dt();
    let t_end = now();
    let hh = h();
    if (li == 0u) {
        var state = b.info.z;
        // It joins at the first step that starts once it has arrived.
        if (state == 0u && st.t_ns - st.dt_ns >= b.info.x) { state = 1u; }
        flying = state;
        let ahead = b.x.xyz + normalize(b.v.xyz + 1e-9) * (b.shape.z + 3.0 * hh);
        let s = sample_pressure((ahead - U.origin.xyz) / hh - 0.5, false);
        p_far = select(0.0, s.x / s.y, s.y > 1e-6);
        // The plane of the rubber it held, facing the air: its normal, and w how far it is
        // from the centre of mass along it. The grid is too coarse to see the air in a
        // pocket of rubber around a bullet, so the rubber has to tell.
        var sn = vec3f(0.0);
        var sx = vec3f(0.0);
        for (var c = 0u; c < 3u; c++) {
            sn[c] = from_fp(atomicExchange(&bullet_hit[bi * HIT_FIELDS + 6u + c], 0), FP_A);
            sx[c] = from_fp(atomicExchange(&bullet_hit[bi * HIT_FIELDS + 9u + c], 0), FP_X);
        }
        let sa = from_fp(atomicExchange(&bullet_hit[bi * HIT_FIELDS + 12u], 0), FP_A);
        held_side = vec4f(0.0);
        if (sa > 0.0 && length(sn) > 0.0) {
            let n = normalize(sn);
            held_side = vec4f(n, dot(sx, n) / sa * 1e-3);
        }
    }
    let state = workgroupUniformLoad(&flying);
    if (state != 1u) {
        if (li == 0u) {
            bullets[bi].info.z = state;
            pose_log[bi * LOG_CAP + st.step % LOG_CAP].x = vec4f(0.0);
        }
        return;
    }
    let pf = workgroupUniformLoad(&p_far);
    let side = workgroupUniformLoad(&held_side);

    // The spin, from the angular momentum and the inertia in the body frame.
    let lb = qrot(vec4f(-b.q.xyz, b.q.w), b.l.xyz);
    let w = qrot(b.q, lb / b.shape.yyx);
    let rho = U.fluid.x;
    var f = vec3f(0.0);
    var tq = vec3f(0.0);
    // Drag of the wetted face, its area across the path, and its centre times that area; w
    // = how far back along the path the face reaches, from the centre of mass.
    var cav = vec4f(0.0);
    var cen = vec4f(0.0, 0.0, 0.0, 3.0e38);
    let vhat = normalize(b.v.xyz + 1e-9);
    for (var k = li; k < b.info.y; k += 256u) {
        let pt = patches[bi * MAX_PATCHES + k];
        let r = qrot(b.q, pt.p.xyz);
        let p = b.x.xyz + r;
        let n = qrot(b.q, pt.n.xyz);
        let area = pt.p.w;
        let X = (p - U.origin.xyz) / hh;
        var uw = vec3f(0.0);
        var sw = vec3f(0.0);
        for (var d = 0u; d < 3u; d++) {
            let s = sample_grid(X - lattice_offset(d), F_U + d, a_wt(d));
            uw[d] = select(0.0, s.x / s.y, s.y > 1e-6);
            sw[d] = s.y;
        }
        var fill = clamp(sample_grid(X - 0.5, F_TYPE, A_WC).y / 8.0, 0.0, 1.0);
        if (in_cavity(bi, p, t_end - step) || dot(r, side.xyz) > side.w) { fill = 0.0; }
        let gs = sample_pressure(X - 0.5, true);
        let p_gas = select(0.0, gs.x / gs.y, gs.y > 1e-6);
        let vp = b.v.xyz + cross(w, r);
        let rel = vp - uw;
        let sp = length(rel);
        let wn = dot(rel, n);
        let q = 0.5 * rho * sp * sp;
        let p_water = max(P_FLOOR, pf + q * cp_patch(wn / max(sp, 1e-9)));
        // The wet share of the patch: pressure, and skin friction along the surface.
        var fw = -fill * p_water * area * n;
        let rt = rel - wn * n;
        let rtl = length(rt);
        if (rtl > 1e-9) { fw -= fill * CF_SKIN * 0.5 * rho * rtl * rt * area; }
        let fp = fw - (1.0 - fill) * max(p_gas, P_FLOOR) * area * n;
        f += fp;
        tq += cross(r, fp);
        for (var d = 0u; d < 3u; d++) {
            push_water(X, d, -fw[d] * step, sw[d]);
        }
        if (wn > 0.0 && fill > 0.0) {
            let face = fill * area * max(dot(n, vhat), 0.0);
            cav += vec4f(-dot(fw, vhat), face, 0.0, 0.0);
            cen = vec4f(cen.xyz + p * face, min(cen.w, dot(r, vhat)));
        }
    }
    red_f[li] = vec4f(f, cav.x);
    red_t[li] = vec4f(tq, cav.y);
    red_c[li] = cen;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (li < off) {
            red_f[li] += red_f[li + off];
            red_t[li] += red_t[li + off];
            red_c[li] = vec4f(red_c[li].xyz + red_c[li + off].xyz, min(red_c[li].w, red_c[li + off].w));
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        // The rubber's pushes came during the last step's substeps, after it was stepped.
        var hit = vec3f(0.0);
        var hit_l = vec3f(0.0);
        for (var c = 0u; c < 3u; c++) {
            hit[c] = from_fp(atomicExchange(&bullet_hit[bi * HIT_FIELDS + c], 0), FP_J);
            hit_l[c] = from_fp(atomicExchange(&bullet_hit[bi * HIT_FIELDS + 3u + c], 0), FP_L);
        }
        let force = red_f[0].xyz + b.x.w * U.gravity.xyz;
        let drag = red_f[0].w;
        let face = red_t[0].w;
        let v = b.v.xyz + (force * step + hit) / b.x.w;
        let x = b.x.xyz + v * step;
        let l = b.l.xyz + red_t[0].xyz * step + hit_l;
        // A symmetric top turns about its angular momentum at |L| / Iy and about its own
        // axis at the rest of its spin. Taking the two exactly keeps the spin and the cone
        // the axis sweeps whatever the step.
        let axis0 = qrot(b.q, vec3f(0.0, 0.0, 1.0));
        let spin_extra = dot(l, axis0) * (1.0 / b.shape.x - 1.0 / b.shape.y) * step;
        let cone = length(l) / b.shape.y * step;
        var q = qmul(b.q, vec4f(0.0, 0.0, sin(0.5 * spin_extra), cos(0.5 * spin_extra)));
        if (cone > 1e-12) {
            q = qmul(vec4f(normalize(l) * sin(0.5 * cone), cos(0.5 * cone)), q);
        }
        q = normalize(q);
        let axis = qrot(q, vec3f(0.0, 0.0, 1.0));
        let roll = b.v.w + dot(l, axis0) / b.shape.x * step;
        var info = b.info;
        info.z = 1u;
        let lo = U.origin.xyz;
        let hi = U.origin.xyz + vec3f(U.dims.xyz) * hh;
        if (any(x < lo) || any(x > hi)) { info.z = 2u; }
        // Out of reach of any water it can go a block per step: it can't miss one on the way,
        // since the blocks around it are checked every step.
        let c = own_cell(x) >> vec3u(2u);
        var near = false;
        for (var k = 0u; k < 27u; k++) {
            let nb = c + vec3i(i32(k / 9u), i32((k / 3u) % 3u), i32(k % 3u)) - 1;
            if (all(nb >= vec3i(0)) && all(nb < vec3i(U.bdims.xyz)) && atomicLoad(&block_ptr[block_slot(nb)]) < PENDING) { near = true; }
        }
        let pace = select(length(v) * U.fluid.y / 4.0, length(v), near);
        if (drag > 0.0 && face > 0.0) {
            // The face's centre, moved back along the path to where the flow leaves it.
            let mid = red_c[0].xyz / face;
            let a = mid + vhat * (red_c[0].w - dot(mid - b.x.xyz, vhat));
            var s: Section;
            s.a = vec4f(a, t_end - 0.5 * step);
            s.b = vec4f(a + v * step, 2.0 * sqrt(drag / (PI * rho * LOGVINOVICH_K)));
            s.c = vec4f(face / PI, 0.0, 0.0, 0.0);
            sections[bi * N_SEC + info.w % N_SEC] = s;
            info.w += 1u;
        }
        var reach = 0.0;
        for (var k = 0u; k < N_SEC; k++) {
            let s = sections[bi * N_SEC + k];
            if (s.b.w <= 0.0 || section_r2(s, t_end) >= R_HAND * R_HAND * hh * hh) { continue; }
            reach = max(reach, max(length(s.a.xyz - x), length(s.b.xyz - x)));
        }
        bullets[bi].x = vec4f(x, b.x.w);
        bullets[bi].v = vec4f(v, roll);
        bullets[bi].q = q;
        bullets[bi].l = vec4f(l, pace);
        bullets[bi].reach = vec4f(x, select(0.0, reach + R_HAND * hh, reach > 0.0));
        bullets[bi].info = info;
        pose_log[bi * LOG_CAP + st.step % LOG_CAP] = PoseEntry(vec4f(x, bitcast<f32>(st.t_ns)), q, vec4f(v, roll));
    }
    storageBarrier();
    workgroupBarrier();
    // Sections thinner than the grid push the water out of the way as they open.
    if (li < N_SEC) {
        let s = sections[bi * N_SEC + li];
        let r2 = section_r2(s, t_end);
        if (s.b.w > 0.0 && r2 < R_HAND * R_HAND * hh * hh) {
            let q = PI * s.b.w * length(s.b.xyz - s.a.xyz);
            let x = (0.5 * (s.a.xyz + s.b.xyz) - U.origin.xyz) / hh - 0.5;
            let base = vec3i(floor(x - 0.5));
            let fx = x - vec3f(base);
            let wx = axis_weights(fx.x);
            let wy = axis_weights(fx.y);
            let wz = axis_weights(fx.z);
            for (var a = 0u; a < 3u; a++) {
                for (var bb = 0u; bb < 3u; bb++) {
                    for (var c = 0u; c < 3u; c++) {
                        let cell = cell_at(base + vec3i(vec3u(a, bb, c)));
                        if (cell == NONE) { continue; }
                        atomicAdd(&acc[acc_i(A_SRC, cell)], to_fp(q * wx[a] * wy[bb] * wz[c], FP_Q));
                    }
                }
            }
        }
    }
}

// Adds the bullets' pushes to the water's faces.
@compute @workgroup_size(64)
fn bullet_apply(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    for (var d = 0u; d < 3u; d++) {
        let du = atomicLoad(&acc[acc_i(A_BU + d, cell)]);
        if (du != 0) { cf[cf_i(F_U + d, cell)] += from_fp(du, FP_M); }
    }
}

// Moves a particle out of any cavity section still thinner than the grid, onto its wall, and
// gives it at least the wall's speed.
fn carve(x: ptr<function, vec3f>, v: ptr<function, vec3f>, id: u32) {
    let t = now();
    let hand2 = R_HAND * R_HAND * h() * h();
    for (var bi = 0u; bi < U.mem.w; bi++) {
        let reach = bullets[bi].reach;
        if (reach.w <= 0.0 || distance(*x, reach.xyz) > reach.w) { continue; }
        for (var k = 0u; k < N_SEC; k++) {
            let s = sections[bi * N_SEC + k];
            let r2 = section_r2(s, t);
            if (s.b.w <= 0.0 || r2 >= hand2) { continue; }
            let ab = s.b.xyz - s.a.xyz;
            let f = dot(*x - s.a.xyz, ab) / max(dot(ab, ab), 1e-20);
            if (f < 0.0 || f > 1.0) { continue; }
            let on = s.a.xyz + f * ab;
            var d = *x - on;
            if (dot(d, d) >= r2) { continue; }
            if (dot(d, d) < 1e-14) {
                var seed = hash3(id, st.step, k);
                d = cross(ab + 1e-9, vec3f(rnd_gauss(&seed), rnd_gauss(&seed), rnd_gauss(&seed)));
            }
            let dir = normalize(d);
            let r = sqrt(r2);
            *x = on + dir * r;
            let wall = s.b.w / (2.0 * r);
            *v += dir * max(wall - dot(*v, dir), 0.0);
        }
    }
}

fn outline_at(bi: u32, k: u32) -> vec2f {
    return outline[bi * OUTLINE + k];
}

struct Near {
    // The nearest point of the outline in the (z, r) plane, the outward normal there, and how
    // far inside the surface the point is (negative outside).
    at: vec2f,
    n: vec2f,
    depth: f32,
}

// The nearest point of bullet `bi`'s surface to `q` = (z, r) of its body frame.
fn nearest_surface(bi: u32, q: vec2f) -> Near {
    let first = outline_at(bi, 0u);
    let last = outline_at(bi, OUTLINE - 1u);
    var o = Near(q, vec2f(0.0), 0.0);
    var best = 3.0e38;
    // The outline, closed by the flat ends where there are any.
    for (var j = 0u; j <= OUTLINE; j++) {
        var p0 = vec2f(first.x, 0.0);
        var p1 = vec2f(last.x, 0.0);
        if (j > 0u) { p0 = outline_at(bi, j - 1u); }
        if (j < OUTLINE) { p1 = outline_at(bi, j); }
        let e = p1 - p0;
        let len2 = dot(e, e);
        if (len2 <= 0.0) { continue; }
        let on = p0 + e * clamp(dot(q - p0, e) / len2, 0.0, 1.0);
        let d = distance(q, on);
        if (d < best) {
            best = d;
            o.at = on;
            o.n = normalize(vec2f(-e.y, e.x));
        }
    }
    var inside = false;
    if (q.x > first.x && q.x < last.x) {
        var k = 1u;
        while (k < OUTLINE - 1u && outline_at(bi, k).x < q.x) { k++; }
        let a = outline_at(bi, k - 1u);
        let c = outline_at(bi, k);
        inside = q.y < mix(a.y, c.y, (q.x - a.x) / max(c.x - a.x, 1e-12));
    }
    o.depth = select(-best, best, inside);
    if (!inside && best > 0.0) { o.n = (q - o.at) / best; }
    return o;
}

// Keeps the rubber around vertex `i` (just moved to `x` at `v`, `vf` the pull of its
// triangles and their shape) out of every bullet in the grid, with the bullets where they are
// at time `t`, the end of a substep of `sub`. Returns the longest substep that still sees a
// bullet coming this fast before it is a quarter of a calibre in.
//
// Rubber of mass `m` slides along a bullet against friction, and the bullet takes the
// opposite. In the air it takes the speed of the surface pushing it too. On water (`wet`) it
// weighs nothing across the surface next to the water behind it, so a bullet only moves it,
// until its own pull `load` presses it onto the bullet. Then the bullet takes that pull
// instead of the water, and the rubber stays on it (`held`) while it presses. Its face is
// half its thickness out from its middle, so it stays held that close, and the drift of a
// substep doesn't let it go.
//
// Stretched far enough, a triangle is wider than a bullet, which would slip through it
// between the corners. So points across each triangle are tried too, as many as keep them a
// radius apart, and the vertex takes its share of what the deepest needs, in proportion to
// its weight there.
fn rubber_contact(i: u32, x: ptr<function, vec3f>, v: ptr<function, vec3f>, m: f32, load: vec3f, vf: VertexForce, wet: bool, held: ptr<function, bool>, t: f32, sub: f32) -> f32 {
    var limit = 3.0e38;
    // Held rubber counts as touching as far out as its face.
    let band = select(0.0, 0.5 * vf.thick, *held);
    *held = false;
    for (var bi = 0u; bi < U.mem.w; bi++) {
        let b = bullets[bi];
        if (b.info.z != 1u) { continue; }
        let xb = b.x.xyz - b.v.xyz * (now() - t);
        let closing = length(b.v.xyz - *v);
        // Near enough to be reached by the next step, which takes its substeps from this one.
        if (length(*x - xb) > b.shape.z + vf.reach + 2.0 * closing * dt()) { continue; }
        limit = min(limit, 0.25 * b.shape.w / max(closing, 1e-6));
        let back = vec4f(-b.q.xyz, b.q.w);
        let w = qrot(b.q, qrot(back, b.l.xyz) / b.shape.yyx);
        // The deepest point found, where on the bullet it goes, the normal there, the
        // vertex's share of it, and its speed against the surface.
        var best = -band;
        var found = false;
        var gap = vec3f(0.0);
        var on_b = vec3f(0.0);
        var nrm = vec3f(0.0);
        var share = 0.0;
        var rel = vec3f(0.0);
        for (var k = vtri_off[i]; k <= vtri_off[i + 1u]; k++) {
            // The last pass is the vertex on its own, which a torn-off one still needs.
            var p = array<vec3f, 3>(*x, *x, *x);
            var pv = array<vec3f, 3>(*v, *v, *v);
            var own = 0u;
            var n = 1u;
            if (k < vtri_off[i + 1u]) {
                let tn = vtri[k];
                if (alive[tn] == 0u) { continue; }
                let tri = tris[tn];
                let ids = array<u32, 3>(tri.v0, tri.v1, tri.v2);
                for (var c = 0u; c < 3u; c++) {
                    if (ids[c] == i) {
                        own = c;
                    } else {
                        pv[c] = mem_v[ids[c]].xyz;
                        p[c] = mem_x[ids[c]].xyz + sub * pv[c];
                    }
                }
                let longest = max(distance(p[0], p[1]), max(distance(p[1], p[2]), distance(p[2], p[0])));
                n = u32(clamp(ceil(longest / b.shape.w), 1.0, 8.0));
            }
            for (var a = 0u; a <= n; a++) {
                for (var c = 0u; c <= n - a; c++) {
                    let bary = vec3f(f32(a), f32(c), f32(n - a - c)) / f32(n);
                    let wi = bary[own];
                    if (wi <= 0.0) { continue; }
                    let s = bary.x * p[0] + bary.y * p[1] + bary.z * p[2];
                    let r = s - xb;
                    if (length(r) >= b.shape.z + band) { continue; }
                    let q = qrot(back, r);
                    let rho = length(q.xy);
                    let near = nearest_surface(bi, vec2f(q.z, rho));
                    if (near.depth <= best) { continue; }
                    let radial = select(vec2f(1.0, 0.0), q.xy / rho, rho > 1e-9);
                    best = near.depth;
                    found = true;
                    on_b = qrot(b.q, vec3f(radial * near.at.y, near.at.x));
                    gap = on_b - r;
                    nrm = qrot(b.q, vec3f(radial * near.n.y, near.n.x));
                    share = wi / dot(bary, bary);
                    rel = bary.x * pv[0] + bary.y * pv[1] + bary.z * pv[2] - b.v.xyz - cross(w, on_b);
                }
            }
        }
        if (!found) { continue; }
        let vn = dot(rel, nrm);
        let rel_t = rel - vn * nrm;
        let slide = length(rel_t);
        // The change in its speed into the surface, the impulse that pushes it on, and the
        // impulse the bullet takes across it.
        var dvn = 0.0;
        var jn = 0.0;
        var j = vec3f(0.0);
        var holds = false;
        if (!wet) {
            dvn = max(-vn, 0.0);
            jn = m * dvn;
            j = -m * share * dvn * nrm;
        } else {
            let ln = dot(load, nrm);
            if (ln < 0.0) {
                dvn = -vn;
                jn = -ln * sub;
                j = min(share, 1.0) * ln * sub * nrm;
                holds = true;
                *held = true;
            } else if (best > 0.0) {
                dvn = max(-vn, 0.0);
            } else {
                continue;
            }
        }
        // Friction on what slides along the surface, over the substep's whole move.
        let stop = select(0.0, min(1.0, MU_JACKET * jn / (m * slide)), slide > 1e-9 && m > 0.0);
        let dvt = -share * stop * rel_t;
        *x += share * gap + sub * dvt;
        *v += share * dvn * nrm + dvt;
        j -= m * dvt;
        let nl = length(vf.n);
        if (holds && nl > 0.0) {
            let a = vf.area * 1e6;
            let sx = (*x - xb) * 1e3 * a;
            for (var c = 0u; c < 3u; c++) {
                atomicAdd(&bullet_hit[bi * HIT_FIELDS + 6u + c], to_fp(vf.n[c] / nl * a, FP_A));
                atomicAdd(&bullet_hit[bi * HIT_FIELDS + 9u + c], to_fp(sx[c], FP_X));
            }
            atomicAdd(&bullet_hit[bi * HIT_FIELDS + 12u], to_fp(a, FP_A));
        }
        let jl = cross(on_b, j);
        for (var c = 0u; c < 3u; c++) {
            atomicAdd(&bullet_hit[bi * HIT_FIELDS + c], to_fp(j[c], FP_J));
            atomicAdd(&bullet_hit[bi * HIT_FIELDS + 3u + c], to_fp(jl[c], FP_L));
        }
    }
    return limit;
}
