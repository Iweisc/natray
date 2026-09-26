// MLS-MPM water, latex membrane with tearing, and spray/mist droplets.
// One `Step` uniform per MPM substep, selected with a dynamic offset.

struct Step {
    t: f32, dt: f32, dt_m: f32, n_particles: u32,
    gravity: vec4f,      // xyz gravity, w = sound speed squared
    grid_origin: vec4f,  // w = dx
    grid_dims: vec4u,    // w = membrane vertex count
    front: array<vec4f, 2>,  // each shot's front (chord coordinate of its leading end) before and after the step
    emit: array<vec4u, 2>,   // spray thrown off at each hole: entry and exit of shot 0, then shot 1, ...
    r_inf: f32, p_cav: f32, tau_vent: f32, v_cap: f32,
    spacing: f32, n_tris: u32, n_emit: u32, spray_cap: u32,
    n_shots: u32, pad0: u32, pad1: u32, pad2: u32,
    mem: vec4f,          // x stiffness scale, y contact friction, z contact mass iso, w damping ratio
    spray: vec4f,        // x mist amount, y wet map half size
    seed: vec4u,         // x step index, y wet map resolution, z unused, w edge count
    misc: vec4f,         // y sheet drag 0.5 rho_air Cd / areal density (1/m)
}

// A bullet's path through the water is tabulated along the chord from where it goes in to
// where it comes out (or stops), src/flight.rs.
struct Shot {
    origin: vec4f,           // where the path enters the water, w = bullet radius
    dir: vec4f,              // chord direction, w = chord length (0 = no water on the path, or not known yet)
    mouth: array<vec4f, 4>,  // entry centre (w cavity strength), entry axis outwards (w when the bullet passed), same for the exit
    info: vec4f,             // x speed at the exit (0 = stopped inside), y/z silhouette radius at entry/exit, w hole radius in the rubber
}

struct PathPoint {
    c: vec3f,     // on the path
    tp: f32,      // when the front passed
    t: vec3f,     // tangent
    k: f32,       // cavity source strength
    mom: vec3f,   // velocity the drag's momentum gives the water on the path
    swirl: f32,   // swirl from the spin: v = swirl r exp(-r^2 / sigma^2)
    foot: vec3f,  // half the bullet's silhouette, along its projected axis
    rb: f32,      // radius of a disk with the silhouette's area
}

const NPATH: u32 = 256u;

struct Particle { pos: vec4f, vel: vec4f, c0: vec4f, c1: vec4f, c2: vec4f }

struct SimState {
    bmin: array<atomic<i32>, 4>,
    bmax: array<atomic<i32>, 4>,
    rmin: vec4i,
    rsize: vec4i,
    spray_count: atomic<u32>,
    pad0: u32, pad1: u32, pad2: u32,
}

struct VInfo { start: u32, count: u32, inv_mass: f32, mass: f32 }
struct Edge { v0: u32, v1: u32, t0: u32, t1: u32, rest: f32, thresh: f32, k: f32, pad: f32 }
struct Tri { v0: u32, v1: u32, v2: u32, rest_area: f32, e0: u32, e1: u32, e2: u32, pad: u32 }
// air.xyz is the air the spray drags along, air.w when it tears loose from the crown.
struct Spray { pos: vec4f, vel: vec4f, air: vec4f }

@group(0) @binding(0) var<uniform> S: Step;
@group(0) @binding(1) var<storage, read_write> P: array<Particle>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> st: SimState;
@group(1) @binding(0) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(4) var<storage, read_write> mp_in: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> mv_in: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> mp_out: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> mv_out: array<vec4f>;
@group(0) @binding(8) var<storage, read> vinfo: array<VInfo>;
@group(0) @binding(9) var<storage, read> adj: array<vec2u>;
@group(0) @binding(10) var<storage, read> edges: array<Edge>;
@group(0) @binding(11) var<storage, read> tris: array<Tri>;
@group(0) @binding(12) var<storage, read_write> alive: array<u32>;
@group(0) @binding(13) var<storage, read_write> spray: array<Spray>;
// When water first touched each ground texel (f32 bits, only ever lowered) and where and
// when each spray slot landed (x, z, radius, t). The renderer rebuilds the wet ground for
// any point in time from these.
@group(0) @binding(14) var<storage, read_write> touch: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read_write> landed: array<vec4f>;
@group(0) @binding(16) var<storage, read> shots: array<Shot>;
// NPATH samples per shot, 4 vec4f each: offset of the path from the chord and tp, tangent
// and k, mom and swirl, foot and rb.
@group(0) @binding(17) var<storage, read> path: array<vec4f>;
// Per particle, how far from each shot's path it was when that bullet passed. 0 until then.
@group(0) @binding(18) var<storage, read_write> hit_r: array<vec4f>;

const FP: f32 = 524288.0;
const NONE: u32 = 0xffffffffu;
const RHO_AIR: f32 = 1.2;
const MU_AIR: f32 = 1.8e-5;
const RHO_W: f32 = 1000.0;
const SIGMA_W: f32 = 0.072;
// Weber number above which a drop is torn apart by the air it moves through.
const WE_CRIT: f32 = 12.0;
const BREAKUP_DROPS: u32 = 4u;

fn node_index(n: vec3u) -> u32 {
    return (n.x * S.grid_dims.y + n.y) * S.grid_dims.z + n.z;
}

fn in_region(n: vec3i) -> bool {
    let lo = st.rmin.xyz;
    let hi = lo + st.rsize.xyz;
    return all(n >= lo) && all(n < hi);
}

fn region_node(i: u32) -> u32 {
    let rs = vec3u(st.rsize.xyz);
    let z = i % rs.z;
    let y = (i / rs.z) % rs.y;
    let x = i / (rs.z * rs.y);
    return node_index(vec3u(st.rmin.xyz) + vec3u(x, y, z));
}

fn ground_touch(p: vec3f) {
    let h = S.spray.y;
    let res = S.seed.y;
    let uv = (p.xz + h) / (2.0 * h);
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return; }
    let c = vec2u(uv * f32(res));
    atomicMin(&touch[c.y * res + c.x], bitcast<u32>(S.t));
}

fn alloc_spray() -> u32 {
    let slot = atomicAdd(&st.spray_count, 1u);
    if (slot >= S.spray_cap) { return NONE; }
    return slot;
}

// Largest drop that survives moving through still air at `speed`.
fn breakup_diameter(speed: f32) -> f32 {
    return WE_CRIT * SIGMA_W / (RHO_AIR * max(speed * speed, 1e-4));
}

// A particle that leaves the resolved flow is a blob of water about one particle spacing
// across, which breaks up below the grid scale. It becomes a few drops with a spread of
// sizes, and whatever volume they don't account for becomes a parcel of finer drops that
// is rendered as a cloud. Fast blobs are shredded by the air into fine spray.
fn break_up(x: vec3f, v: vec3f, id: u32) {
    let first = atomicAdd(&st.spray_count, BREAKUP_DROPS + 1u);
    if (first >= S.spray_cap) { return; }
    var seed = hash3(S.seed.x, id, 0xb4eau);
    let h = S.spacing;
    let vol = h * h * h;
    let speed = length(v);
    // Ligaments at the grid scale pinch off into drops about half its size.
    let d_med = min(0.55 * h, 0.5 * breakup_diameter(speed));
    var d: array<f32, BREAKUP_DROPS>;
    var used = 0.0;
    for (var k = 0u; k < BREAKUP_DROPS; k++) {
        d[k] = clamp(d_med * exp(0.5 * rnd_gauss(&seed)), 2e-5, 1.24 * h);
        used += 0.5236 * d[k] * d[k] * d[k];
    }
    let df = 0.5 * d_med;
    // Only drops too small to see one by one belong in a parcel. Slower blobs put their
    // whole volume into the visible drops.
    let fine = df < 1e-4;
    let scale = select(pow(vol / used, 1.0 / 3.0), min(1.0, pow(vol / used, 1.0 / 3.0)), fine);
    for (var k = 0u; k < BREAKUP_DROPS; k++) {
        let slot = first + k;
        if (slot >= S.spray_cap) { return; }
        let dk = d[k] * scale;
        // Fragments separate at about the capillary speed, plus a share of the airflow.
        let spread = sqrt(SIGMA_W / (RHO_W * dk)) + 0.05 * speed;
        let u = normalize(vec3f(rnd_gauss(&seed), rnd_gauss(&seed), rnd_gauss(&seed)) + 1e-6);
        let off = (vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5) * h;
        spray[slot].pos = vec4f(x + off, 0.5 * dk);
        spray[slot].vel = vec4f(v + u * spread * rnd(&seed), 1.0);
        spray[slot].air = vec4f(v * 0.2, S.t);
    }
    let slot = first + BREAKUP_DROPS;
    if (slot >= S.spray_cap) { return; }
    let n = (vol - used * scale * scale * scale) / (0.5236 * df * df * df);
    // Weight 1 means a single drop, so a parcel needs at least two.
    spray[slot].pos = vec4f(x, select(0.0, 0.5 * df, fine && n >= 2.0));
    spray[slot].vel = vec4f(v, max(n, 2.0));
    spray[slot].air = vec4f(v * 0.3, S.t);
}

@compute @workgroup_size(1)
fn region_prep() {
    let d = vec3i(S.grid_dims.xyz);
    let bmin = vec3i(atomicLoad(&st.bmin[0]), atomicLoad(&st.bmin[1]), atomicLoad(&st.bmin[2]));
    let bmax = vec3i(atomicLoad(&st.bmax[0]), atomicLoad(&st.bmax[1]), atomicLoad(&st.bmax[2]));
    var lo = clamp(bmin - 2, vec3i(0), d - 1);
    var hi = clamp(bmax + 5, lo + 1, d);
    if (bmax.x < bmin.x) { lo = vec3i(0); hi = vec3i(1); }
    st.rmin = vec4i(lo, 0);
    st.rsize = vec4i(hi - lo, 0);
    let n = u32((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z));
    indirect[0] = (n + 255u) / 256u;
    indirect[1] = 1u;
    indirect[2] = 1u;
    let ns = min(atomicLoad(&st.spray_count), S.spray_cap);
    let g = (ns + 255u) / 256u;
    indirect[3] = min(g, 65535u);
    indirect[4] = (g + 65534u) / 65535u;
    indirect[5] = 1u;
    for (var k = 0; k < 3; k++) {
        atomicStore(&st.bmin[k], 1 << 30);
        atomicStore(&st.bmax[k], -(1 << 30));
    }
}

@compute @workgroup_size(256)
fn grid_clear(@builtin(global_invocation_id) gid: vec3u) {
    let rs = st.rsize.xyz;
    if (gid.x >= u32(rs.x * rs.y * rs.z)) { return; }
    let n = region_node(gid.x) * 4u;
    atomicStore(&grid[n], 0);
    atomicStore(&grid[n + 1u], 0);
    atomicStore(&grid[n + 2u], 0);
    atomicStore(&grid[n + 3u], 0);
}

fn weights(fx: vec3f) -> array<vec3f, 3> {
    return array<vec3f, 3>(
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5)
    );
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= S.n_particles) { return; }
    let p = P[i];
    if (p.vel.w < 0.5) { return; }
    let dx = S.grid_origin.w;
    let inv_dx = 1.0 / dx;
    let xp = (p.pos.xyz - S.grid_origin.xyz) * inv_dx;
    let base = vec3i(floor(xp - 0.5));
    let fx = xp - vec3f(base);
    let w = weights(fx);
    // Weakly compressible water. g2p lets J exceed 1 so the water can hold up to -1 atm.
    let stress = -S.dt * 4.0 * inv_dx * inv_dx * S.gravity.w * (p.pos.w - 1.0);
    let affine = mat3x3f(vec3f(stress, 0.0, 0.0), vec3f(0.0, stress, 0.0), vec3f(0.0, 0.0, stress))
        + mat3x3f(p.c0.xyz, p.c1.xyz, p.c2.xyz);
    for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
            for (var c = 0; c < 3; c++) {
                let off = vec3i(a, b, c);
                let dpos = (vec3f(off) - fx) * dx;
                let weight = w[a].x * w[b].y * w[c].z;
                let mom = weight * (p.vel.xyz + affine * dpos);
                let idx = node_index(vec3u(base + off)) * 4u;
                atomicAdd(&grid[idx], i32(round(mom.x * FP)));
                atomicAdd(&grid[idx + 1u], i32(round(mom.y * FP)));
                atomicAdd(&grid[idx + 2u], i32(round(mom.z * FP)));
                atomicAdd(&grid[idx + 3u], i32(round(weight * FP)));
            }
        }
    }
}

@compute @workgroup_size(256)
fn grid_update(@builtin(global_invocation_id) gid: vec3u) {
    let rs = st.rsize.xyz;
    if (gid.x >= u32(rs.x * rs.y * rs.z)) { return; }
    let node = region_node(gid.x);
    let idx = node * 4u;
    let m = f32(atomicLoad(&grid[idx + 3u])) / FP;
    var v = vec3f(0.0);
    if (m > 1e-6) {
        v = vec3f(f32(atomicLoad(&grid[idx])), f32(atomicLoad(&grid[idx + 1u])), f32(atomicLoad(&grid[idx + 2u]))) / (FP * m);
        v += S.dt * S.gravity.xyz;
        let rs3 = vec3u(rs);
        let local = vec3u(gid.x / (rs3.z * rs3.y), (gid.x / rs3.z) % rs3.y, gid.x % rs3.z);
        let g = vec3u(st.rmin.xyz) + local;
        let wy = S.grid_origin.y + f32(g.y) * S.grid_origin.w;
        // Ground: no penetration. A spreading film is held back by viscosity, which is weak
        // until it is thinner than the particles, not by dry friction. Coulomb friction here
        // pinned the puddle's edge into a grid-aligned square.
        if (wy <= 0.0 && v.y < 0.0) { v.y = 0.0; }
        if (g.y >= S.grid_dims.y - 3u) { v.y = min(v.y, 0.0); }
    }
    atomicStore(&grid[idx], bitcast<i32>(v.x));
    atomicStore(&grid[idx + 1u], bitcast<i32>(v.y));
    atomicStore(&grid[idx + 2u], bitcast<i32>(v.z));
}

fn path_at(s: u32, a: f32) -> PathPoint {
    let sh = shots[s];
    let f = clamp(a / sh.dir.w, 0.0, 1.0) * f32(NPATH - 1u);
    let i = min(u32(f), NPATH - 2u);
    let w = f - f32(i);
    let b = (s * NPATH + i) * 4u;
    let p0 = mix(path[b], path[b + 4u], w);
    let p1 = mix(path[b + 1u], path[b + 5u], w);
    let p2 = mix(path[b + 2u], path[b + 6u], w);
    let p3 = mix(path[b + 3u], path[b + 7u], w);
    return PathPoint(sh.origin.xyz + sh.dir.xyz * a + p0.xyz, p0.w, normalize(p1.xyz), p1.w, p2.xyz, p2.w, p3.xyz, p3.w);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= S.n_particles) { return; }
    var p = P[i];
    if (p.vel.w < 0.5) { return; }
    let dx = S.grid_origin.w;
    let inv_dx = 1.0 / dx;
    let xp = (p.pos.xyz - S.grid_origin.xyz) * inv_dx;
    let base = vec3i(floor(xp - 0.5));
    let fx = xp - vec3f(base);
    let w = weights(fx);
    var v = vec3f(0.0);
    var b0 = vec3f(0.0);
    var b1 = vec3f(0.0);
    var b2 = vec3f(0.0);
    var m_near = 0.0;
    for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
            for (var c = 0; c < 3; c++) {
                let off = vec3i(a, b, c);
                let dpos = (vec3f(off) - fx) * dx;
                let weight = w[a].x * w[b].y * w[c].z;
                let idx = node_index(vec3u(base + off)) * 4u;
                let gv = vec3f(bitcast<f32>(atomicLoad(&grid[idx])), bitcast<f32>(atomicLoad(&grid[idx + 1u])), bitcast<f32>(atomicLoad(&grid[idx + 2u])));
                m_near += weight * f32(atomicLoad(&grid[idx + 3u])) / FP;
                v += weight * gv;
                b0 += weight * gv * dpos.x;
                b1 += weight * gv * dpos.y;
                b2 += weight * gv * dpos.z;
            }
        }
    }
    let cs = 4.0 * inv_dx * inv_dx;
    var c0 = b0 * cs;
    var c1 = b1 * cs;
    var c2 = b2 * cs;
    var x = p.pos.xyz;

    // Bullets: when one's front passes this particle's slab, the water gets the kinetic
    // energy of a cylindrical cavity (source flow, v = K/r), the momentum of the drag and the
    // swirl of the spin. Behind it the cavity is near vacuum until air vents in, so the
    // atmosphere decelerates the expansion and then collapses it. The water is
    // incompressible, so the cavity in this slab is open exactly while the particle is
    // further out than it started.
    var hit = hit_r[i];
    let hit_before = hit;
    for (var s = 0u; s < S.n_shots; s++) {
        let sh = shots[s];
        let len = sh.dir.w;
        let ax0 = dot(x - sh.origin.xyz, sh.dir.xyz);
        // Water off the ends of the path is pushed too, once the front reaches that end.
        let ac = clamp(ax0, 0.0, len);
        let passing = ac > S.front[0][s] && ac <= S.front[1][s];
        if (len <= 0.0 || (hit[s] == 0.0 && !passing)) { continue; }
        // One Newton step to the point of the path nearest the particle, since the path bends
        // once the bullet yaws.
        var q = path_at(s, ax0);
        q = path_at(s, clamp(ax0 + dot(x - q.c, q.t) * dot(q.t, sh.dir.xyz), 0.0, len));
        let tau = S.t - q.tp;
        if (tau > 8.0 * S.tau_vent) { continue; }
        let rel = x - q.c;
        let along = dot(rel, q.t);
        let radial = rel - along * q.t;
        let r = length(radial);
        let rdir = select(ortho_basis(q.t)[0], radial / max(r, 1e-9), r > 1e-7);
        // The cavity is a line source only as long as the path through the water. Its flow
        // has K / r across the middle, half that across the end planes, and bends round the
        // ends, out through the holes and dying away past them.
        let xa = select(ax0, along + select(0.0, len, ax0 > len), ax0 < 0.0 || ax0 > len);
        let rr = max(r, q.rb);
        let d0 = inverseSqrt(xa * xa + rr * rr);
        let d1 = inverseSqrt((xa - len) * (xa - len) + rr * rr);
        let flow = 0.5 * (rdir * (xa * d0 - (xa - len) * d1) + q.t * rr * (d1 - d0));
        if (hit[s] == 0.0) {
            hit[s] = rr;
            let sig = 2.5 * q.rb;
            let g = exp(-(r * r) / (sig * sig));
            v += flow * (q.k / rr) + q.mom * g + cross(q.t, rdir) * q.swirl * r * g;
            if (r < q.rb && ax0 >= 0.0 && ax0 <= len) { x += rdir * (q.rb - r); }
        } else if (tau > 0.0 && r > hit[s]) {
            let pc = S.p_cav * exp(-tau / S.tau_vent);
            v -= flow * (pc / (RHO_W * max(r, 0.01) * 1.5) * S.dt);
        }
        // The grid is too coarse for the K / r flow within a cell or so of the path: water on
        // opposite sides shares nodes and its momentum cancels, so the cavity barely opened.
        // While the analytic cavity grows, its wall pushes the water out like a piston.
        let cav = cavity(q.k, tau, q.rb, S.p_cav, S.tau_vent);
        if (tau > 0.0 && cav.y > 0.0 && r < cav.x && ax0 >= 0.0 && ax0 <= len) {
            x += rdir * (cav.x - r);
            v += rdir * max(cav.y - dot(v, rdir), 0.0);
        }
    }
    if (any(hit != hit_before)) { hit_r[i] = hit; }

    let speed = length(v);
    x += S.dt * v;
    var J = p.pos.w * (1.0 + S.dt * (c0.x + c1.y + c2.z));
    // Tension is allowed down to -1 atm: a void opening inside the water would be vacuum,
    // so the surrounding air pressure holds the water together up to that point.
    J = clamp(J, 0.6, 1.0 + 101325.0 / (RHO_W * S.gravity.w));

    let lo = S.grid_origin.xyz + 3.5 * dx;
    let hi = S.grid_origin.xyz + (vec3f(S.grid_dims.xyz) - 4.5) * dx;
    let escaped = any(x.xz < lo.xz) || any(x.xz > hi.xz) || x.y > hi.y;
    // 1 inside bulk water, about 0.3 in a sheet one particle thick, less once that sheet
    // stretches. Sheets and threads that thin break up into drops in reality, and neither
    // the grid nor the renderer can resolve them anyway.
    let fill = m_near * pow(S.spacing / dx, 3.0);
    // Jets and sheets a particle or two thick are unstable in the air: jets neck into beads
    // (Rayleigh-Plateau) and sheets tear open, within about three capillary times
    // 3 sqrt(rho r^3 / sigma) of a thread this thin. Water resting on the ground is a puddle.
    let r_thin = 0.75 * S.spacing;
    let t_thin = 3.0 * sqrt(RHO_W * r_thin * r_thin * r_thin / SIGMA_W);
    let thin_age = select(0.0, p.c1.w + S.dt, fill < 0.45 && x.y > 0.02);
    if (speed > S.v_cap || escaped || fill < 0.2 || thin_age > t_thin) {
        // Too fast for the grid, leaving the domain, isolated or a thin structure that has
        // lived out its breakup time: it becomes drops.
        if (x.y < 1.5 * S.spacing) {
            ground_touch(x);
            // A film still racing outward lifts off at its rim and throws drops, which is the
            // crown of the splash. A slow one just soaks in where it is.
            let vh = length(v.xz);
            if (vh > 1.0) {
                var seed = hash3(S.seed.x, i, 0x5a11u);
                let lift = vh * tan(mix(0.1, 0.45, rnd(&seed)));
                break_up(x + vec3f(0.0, S.spacing, 0.0), vec3f(v.x, max(v.y, 0.0) + lift, v.z), i);
            }
        } else {
            break_up(x, v, i);
        }
        P[i].vel = vec4f(v, 0.0);
        return;
    }
    let floor_y = 0.3 * S.spacing;
    if (x.y < floor_y) {
        x.y = floor_y;
        v.y = max(v.y, 0.0);
    }
    if (x.y < 0.7 * S.spacing) { ground_touch(x); }
    // Water that has come to rest on the ground is a puddle only a particle or two deep,
    // which the particles would render as rippled glass. The flat wet map draws it instead.
    if (x.y < 1.5 * S.spacing && speed < 0.3) {
        ground_touch(x);
        P[i].vel = vec4f(v, 0.0);
        return;
    }

    P[i].pos = vec4f(x, J);
    P[i].vel = vec4f(v, 1.0);
    P[i].c0 = vec4f(c0, 0.0);
    P[i].c1 = vec4f(c1, thin_age);
    P[i].c2 = vec4f(c2, 0.0);
    let cell = vec3i(floor((x - S.grid_origin.xyz) * inv_dx - 0.5));
    atomicMin(&st.bmin[0], cell.x);
    atomicMin(&st.bmin[1], cell.y);
    atomicMin(&st.bmin[2], cell.z);
    atomicMax(&st.bmax[0], cell.x);
    atomicMax(&st.bmax[1], cell.y);
    atomicMax(&st.bmax[2], cell.z);
}

// ---------------------------------------------------------------- membrane

fn node_mass(n: vec3i) -> f32 {
    if (!in_region(n)) { return 0.0; }
    return f32(atomicLoad(&grid[node_index(vec3u(n)) * 4u + 3u])) / FP;
}

fn node_vel(n: vec3i) -> vec3f {
    if (!in_region(n)) { return vec3f(0.0); }
    let idx = node_index(vec3u(n)) * 4u;
    return vec3f(bitcast<f32>(atomicLoad(&grid[idx])), bitcast<f32>(atomicLoad(&grid[idx + 1u])), bitcast<f32>(atomicLoad(&grid[idx + 2u])));
}

// Node mass and its gradient under the same quadratic B-spline as the transfers. Trilinear
// sampling kinks at every cell face, and the taut rubber resting on it showed those kinks
// as dents all over the balloon.
fn mass_grad(p: vec3f) -> vec4f {
    let inv_dx = 1.0 / S.grid_origin.w;
    let xp = (p - S.grid_origin.xyz) * inv_dx;
    let base = vec3i(floor(xp - 0.5));
    let fx = xp - vec3f(base);
    let w = weights(fx);
    let dw = array<vec3f, 3>(fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5);
    var mg = vec4f(0.0);
    for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
            for (var c = 0; c < 3; c++) {
                let m = node_mass(base + vec3i(a, b, c));
                mg += m * vec4f(
                    dw[a].x * w[b].y * w[c].z * inv_dx,
                    w[a].x * dw[b].y * w[c].z * inv_dx,
                    w[a].x * w[b].y * dw[c].z * inv_dx,
                    w[a].x * w[b].y * w[c].z);
            }
        }
    }
    return mg;
}

fn sample_water_vel(p: vec3f) -> vec3f {
    let g = (p - S.grid_origin.xyz) / S.grid_origin.w;
    let i0 = vec3i(floor(g));
    let f = g - vec3f(i0);
    var v = vec3f(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let wv = select(1.0 - f, f, vec3<bool>(o.x == 1, o.y == 1, o.z == 1));
        let m = node_mass(i0 + o);
        let ww = wv.x * wv.y * wv.z * m;
        v += ww * node_vel(i0 + o);
        wsum += ww;
    }
    return v / max(wsum, 1e-6);
}

fn edge_alive(e: Edge) -> bool {
    return alive[e.t0] != 0u || (e.t1 != NONE && alive[e.t1] != 0u);
}

@compute @workgroup_size(256)
fn mem_step(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= S.grid_dims.w) { return; }
    let info = vinfo[i];
    let x = mp_in[i].xyz;
    let v = mv_in[i].xyz;
    if (info.inv_mass == 0.0) {
        mp_out[i] = mp_in[i];
        mv_out[i] = vec4f(0.0);
        return;
    }
    let ks = S.mem.x;
    var f = vec3f(0.0);
    var n_alive = 0u;
    for (var e = info.start; e < info.start + info.count; e++) {
        let a = adj[e];
        let ed = edges[a.y];
        if (!edge_alive(ed)) { continue; }
        n_alive++;
        let xn = mp_in[a.x].xyz;
        let d = xn - x;
        let len = length(d);
        if (len < 1e-9) { continue; }
        let dirn = d / len;
        let stretch = len / ed.rest;
        let kk = ed.k * ks;
        // Rubber membranes carry no compression; slack springs let it crumple.
        if (stretch > 1.0) { f += kk * (stretch - 1.0) * dirn; }
        let damp = S.mem.w * 2.0 * sqrt(kk / ed.rest * info.mass);
        f += damp * dot(mv_in[a.x].xyz - v, dirn) * dirn;
    }
    var acc = f * info.inv_mass + S.gravity.xyz;
    // One grid lookup per substep serves both the support and the contact below. A vertex
    // moves a fraction of a millimetre per substep, so the contact extrapolates from it.
    var mg = vec4f(0.0);
    if (n_alive > 0u) {
        mg = mass_grad(x);
        // Water is incompressible, so where the rubber rests on it the water takes the inward
        // pull of the tension. Without this the rubber sinks into the grid's mass field until
        // the contact below catches it, and that field's lumps show up as dents.
        let gl = length(mg.xyz);
        if (mg.w > 0.5 * S.mem.z && gl > 1e-6) {
            let n = -mg.xyz / gl;
            acc -= min(dot(acc, n), 0.0) * n;
        }
    }
    var vv = v + S.dt_m * acc;
    // Quadratic air drag on the sheet, implicit so fast scraps stay stable.
    vv /= 1.0 + S.dt_m * S.misc.y * length(vv);
    var xx = x + S.dt_m * vv;

    if (n_alive > 0u) {
        // Contact with water: the rubber cannot sink into it, and moving water pushes it.
        let iso = S.mem.z;
        let m = mg.w + dot(mg.xyz, xx - x);
        if (m > iso) {
            let gm = mg.xyz;
            let gl = length(gm);
            if (gl > 1e-6) {
                let n = -gm / gl;
                let depth = (m - iso) / gl;
                xx += n * min(depth, S.grid_origin.w);
                let vw = sample_water_vel(xx);
                let rel = vv - vw;
                let vn = dot(rel, n);
                if (vn < 0.0) {
                    var vt = rel - vn * n;
                    vt *= max(0.0, 1.0 - S.mem.y * (-vn) / max(length(vt), 1e-6));
                    vv = vw + vt;
                }
            }
        }
    }
    if (xx.y < 0.0005) {
        xx.y = 0.0005;
        vv = vec3f(vv.x * 0.9, max(vv.y, 0.0), vv.z * 0.9);
    }
    mp_out[i] = vec4f(xx, mp_in[i].w);
    mv_out[i] = vec4f(vv, 0.0);
}

@compute @workgroup_size(256)
fn mem_break(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= S.n_tris || alive[t] == 0u) { return; }
    let tri = tris[t];
    let p0 = mp_in[tri.v0].xyz;
    let p1 = mp_in[tri.v1].xyz;
    let p2 = mp_in[tri.v2].xyz;
    let c = (p0 + p1 + p2) / 3.0;
    // Holed wherever a bullet's silhouette has gone through: a capsule along its projected
    // axis, which tears a keyhole once it yaws.
    for (var s = 0u; s < S.n_shots; s++) {
        let sh = shots[s];
        let ax = dot(c - sh.origin.xyz, sh.dir.xyz);
        if (sh.dir.w <= 0.0 || ax > S.front[1][s]) { continue; }
        let q = path_at(s, ax);
        var rel = c - q.c;
        rel -= dot(rel, q.t) * q.t;
        let h = clamp(dot(rel, q.foot) / max(dot(q.foot, q.foot), 1e-12), -1.0, 1.0);
        if (length(rel - h * q.foot) < sh.info.w) {
            alive[t] = 0u;
            return;
        }
    }
    let ids = array<u32, 3>(tri.e0, tri.e1, tri.e2);
    for (var k = 0; k < 3; k++) {
        let e = edges[ids[k]];
        let len = length(mp_in[e.v0].xyz - mp_in[e.v1].xyz);
        if (len > e.thresh * e.rest) {
            alive[t] = 0u;
            return;
        }
    }
}

// ---------------------------------------------------------------- spray

// Spray thrown off the cavity lip where the cavity opens through the entry and exit holes.
// The lip throws off a sheet, the crown, that tears into drops a little later. A drop starts
// on the lip and rides its slice of the sheet until that tears. Until then it is part of
// the sheet, which the renderer draws from the same formulas (lip.wgsl). At the exit the
// water the bullet pushes ahead and drags along leaves first, close to the path. Drop size
// is capped by aerodynamic breakup, so the fast ejecta are fine spray.
@compute @workgroup_size(256)
fn spray_emit(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= S.n_emit) { return; }
    var seed = hash3(S.seed.x, i, 0x5eedu);
    // Hole m is shot m / 2, entry or exit, the same numbering as the crown's.
    var mouth = 0u;
    var first = 0u;
    for (; mouth < 7u; mouth++) {
        let n = S.emit[mouth / 4u][mouth % 4u];
        if (i < first + n) { break; }
        first += n;
    }
    let sh = shots[mouth / 2u];
    let is_exit = (mouth & 1u) == 1u;
    let e = select(0u, 2u, is_exit);
    let center = sh.mouth[e].xyz;
    let axis = sh.mouth[e + 1u].xyz;
    let basis = ortho_basis(axis);

    // Spread over the step instead of releasing everything at its end.
    let t_emit = S.t - S.dt * rnd(&seed);
    let tau = max(t_emit - sh.mouth[e + 1u].w, 0.0);
    let rb = select(sh.info.y, sh.info.z, is_exit);
    let cav = cavity(sh.mouth[e].w, tau, rb, S.p_cav, S.tau_vent);
    let lv = lip_velocity(cav.y);
    var phi = 2.0 * PI * rnd(&seed);
    if (rnd(&seed) < 0.65) {
        let f = finger_at(mouth, u32(rnd(&seed) * f32(LIP_FINGERS)) % LIP_FINGERS);
        phi = fract((f.x + 0.35 * f.z * rnd_gauss(&seed)) / (2.0 * PI)) * 2.0 * PI;
    }
    let out = basis * vec3f(cos(phi), sin(phi), 0.0);
    var pos = center + out * cav.x - axis * lip_sag(cav.x, S.r_inf);
    var vel = out * lv.x + axis * lv.y;
    var release = t_emit + sheet_life(length(vel)) * (1.0 + 0.6 * finger(phi, mouth));
    if (is_exit && rnd(&seed) < exp(-tau / 3e-5)) {
        // The water pushed ahead of the bullet: a narrow, fast column, loose from the start.
        let theta = 0.14 * rnd(&seed);
        pos = center + out * rb * sqrt(rnd(&seed));
        vel = (out * sin(theta) + axis * cos(theta)) * sh.info.x * mix(0.1, 0.6, rnd(&seed));
        release = t_emit;
    }
    pos += vel * (S.t - t_emit);
    let speed = length(vel);

    let slot = alloc_spray();
    if (slot == NONE) { return; }
    // The sheet thins as it stretches, and the fast early sheets stretch most.
    let d_sheet = clamp(6e-4 * sqrt(10.0 / max(speed, 1.0)), 1.5e-4, 1.5e-3);
    let dia = min(d_sheet * exp(0.6 * rnd_gauss(&seed)), 0.5 * breakup_diameter(speed) * exp(0.35 * rnd_gauss(&seed)));
    let r_drop = 0.5 * clamp(dia, 2e-5, 2.5e-3);
    var r: f32;
    var weight: f32;
    var air: vec3f;
    if (rnd(&seed) < mist_share(speed)) {
        // Every emission stands for the same piece of lip material, so a parcel carries the
        // mass of the drop it replaced, shredded into droplets that small. S.spray.x is the
        // mist slider, not a mass budget.
        r = mix(5e-6, 25e-6, rnd(&seed));
        weight = S.spray.x * pow(r_drop / r, 3.0);
        air = vel * 0.55;
    } else {
        r = r_drop;
        weight = 1.0;
        air = vel * 0.25;
    }
    spray[slot].pos = vec4f(pos, r);
    spray[slot].vel = vec4f(vel, weight);
    spray[slot].air = vec4f(air, release);
}

fn drag_cd(re: f32) -> f32 {
    if (re < 1e-4) { return 24.0 / 1e-4; }
    return 24.0 / re * (1.0 + 0.15 * pow(re, 0.687)) + 0.42 / (1.0 + 42500.0 * pow(re, -1.16));
}

@compute @workgroup_size(256)
fn spray_update(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= min(atomicLoad(&st.spray_count), S.spray_cap)) { return; }
    var s = spray[i];
    var r = s.pos.w;
    if (r <= 0.0) { return; }
    let dt = S.dt;
    // air.w is when the drop tears loose from the crown sheet. Until then it moves straight
    // on with its slice of the sheet, as the renderer draws it.
    let release = s.air.w;
    var ua = s.air.xyz;
    var v = s.vel.xyz;
    if (S.t >= release) {
        var seed = hash3(S.seed.x, i, 0xa11u);
        ua *= exp(-dt / 0.035);
        if (s.vel.w > 1.0) {
            // Mist follows small eddies in the air it dragged along.
            let kick = vec3f(rnd_gauss(&seed), rnd_gauss(&seed), rnd_gauss(&seed));
            ua += kick * 0.35 * sqrt(dt);
        } else if (S.t - dt < release) {
            // Fragments of a torn sheet separate at about the capillary speed, and the
            // flapping sheet throws them a few degrees off its own direction.
            let u = normalize(vec3f(rnd_gauss(&seed), rnd_gauss(&seed), rnd_gauss(&seed)) + 1e-6);
            v += u * (sqrt(SIGMA_W / (RHO_W * 2.0 * r)) + 0.08 * length(v)) * rnd(&seed);
        }
        let rel = v - ua;
        let sp = length(rel);
        let re = 2.0 * r * sp * RHO_AIR / MU_AIR;
        let k = 3.0 * RHO_AIR * drag_cd(re) * sp / (8.0 * RHO_W * r);
        // Implicit drag keeps tiny droplets stable at any dt.
        v = (v + dt * (S.gravity.xyz + k * ua)) / (1.0 + k * dt);
        if (s.vel.w > 1.0) {
            // Evaporation of mist droplets (d(r^2)/dt roughly constant).
            r = sqrt(max(r * r - 1.2e-10 * dt, 0.0));
            if (r < 2.5e-6) { r = 0.0; }
        }
    }
    let pos = s.pos.xyz + dt * v;
    if (pos.y <= r) {
        // A parcel lands with the volume of all its drops.
        landed[i] = vec4f(pos.x, pos.z, r * pow(s.vel.w, 1.0 / 3.0), S.t);
        r = 0.0;
    }
    if (length(pos.xz) > 4.0 || S.t > release + 6.0) { r = 0.0; }
    spray[i].pos = vec4f(pos, r);
    spray[i].vel = vec4f(v, s.vel.w);
    spray[i].air = vec4f(ua, release);
}
