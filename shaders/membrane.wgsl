// The rubber: a triangle mesh of Ogden latex, stepped explicitly in substeps of the water's
// step, with the mesh of the uninflated balloon as its rest shape.
//
// Where it lies on water it weighs next to nothing against the water behind it, so it goes
// wherever the water surface goes. Its pull across the surface is what the water feels
// instead: mem_splat hands it to the grid as the pressure of the air cells behind it, which
// the water's pressure solve takes as given. Along the surface it slides, held back by the
// thin layer of water it drags along, far thinner than a cell. Where its pull presses it onto
// a bullet, the bullet holds it and takes the pull instead, see rubber_contact.

struct Tri { v0: u32, v1: u32, v2: u32, rest_area: f32, dm_inv: vec4f }

// Read from here, and written to the other buffer by each substep. The two swap every
// substep, and a step has an even number of them. mem_v.w is 1 where a bullet holds it.
@group(0) @binding(15) var<storage, read_write> mem_x: array<vec4f>;
@group(0) @binding(16) var<storage, read_write> mem_v: array<vec4f>;
@group(0) @binding(17) var<storage, read_write> mem_x_out: array<vec4f>;
@group(0) @binding(18) var<storage, read_write> mem_v_out: array<vec4f>;
@group(0) @binding(19) var<storage, read_write> tris: array<Tri>;
// Each vertex's triangles.
@group(0) @binding(20) var<storage, read_write> vtri_off: array<u32>;
@group(0) @binding(21) var<storage, read_write> vtri: array<u32>;
// 0 where the knot holds it.
@group(0) @binding(22) var<storage, read_write> mem_mass: array<f32>;
// The water where each vertex is: its velocity, and how full of water the grid is there,
// 1 inside and 0 away from it.
@group(0) @binding(23) var<storage, read_write> mem_water: array<vec4f>;
@group(0) @binding(24) var<storage, read_write> alive: array<u32>;
// LAYER_MODES per vertex, see drag_modes.
@group(0) @binding(25) var<storage, read_write> mem_layer: array<vec4f>;
// Each vertex's push on the water (N) and its area (m^2), from the last substep. The push is
// 0 off the water, and where a bullet holds it.
@group(0) @binding(26) var<storage, read_write> mem_push: array<vec2f>;

const OGDEN_ALPHA: vec3f = vec3f(1.3, 5.0, -2.0);
// Rubber this deep in water is on it. The rubber sits about half a particle spacing out
// from the outermost particles.
const CONTACT: f32 = 0.05;

struct Sheet {
    // Cauchy stresses along the major and minor stretch.
    sig: vec2f,
    // How fast the nominal stress along the major stretch grows with it, for the stable step.
    tangent: f32,
}

// Rubber can't be squeezed in its plane: it wrinkles instead, which leaves it pulled one way
// only, or slack. Each stretch's powers come from one log, since this is most of the cost
// of a substep.
fn sheet(major: f32, minor: f32) -> Sheet {
    if (major <= 1.0) { return Sheet(vec2f(0.0), 0.0); }
    let mu = U.rubber.xyz;
    let la = log(major);
    let pa = exp(OGDEN_ALPHA * la);
    let tan_scale = 1.0 / (major * major);
    if (minor * minor * major < 1.0) {
        // Pulled one way, it narrows as it would in a strip.
        let pn = exp(-0.5 * OGDEN_ALPHA * la);
        return Sheet(vec2f(dot(mu, pa - pn), 0.0), dot(mu * OGDEN_ALPHA, pa + 0.5 * pn) * tan_scale);
    }
    let lb = log(minor);
    let pb = exp(OGDEN_ALPHA * lb);
    let p3 = exp(-OGDEN_ALPHA * (la + lb));
    return Sheet(vec2f(dot(mu, pa - p3), dot(mu, pb - p3)), dot(mu * OGDEN_ALPHA, pa + p3) * tan_scale);
}

struct VertexForce {
    // The rubber's pull on the vertex.
    f: vec3f,
    // Outward, twice the area of the triangles around it.
    n: vec3f,
    // The vertex's share of that area.
    area: f32,
    // Stiffness against moving the vertex, for the stable step.
    stiff: f32,
    // The largest tension in the triangles around it.
    tension: f32,
    // How far they reach from it.
    reach: f32,
    // How thick they are.
    thick: f32,
}

fn vertex_force(i: u32) -> VertexForce {
    var o = VertexForce(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0, U.rubber.w);
    var rest = 0.0;
    for (var k = vtri_off[i]; k < vtri_off[i + 1u]; k++) {
        let t = vtri[k];
        if (alive[t] == 0u) { continue; }
        let tri = tris[t];
        let x0 = mem_x[tri.v0].xyz;
        let e1 = mem_x[tri.v1].xyz - x0;
        let e2 = mem_x[tri.v2].xyz - x0;
        let m = tri.dm_inv;
        o.reach = max(o.reach, max(length(e1), max(length(e2), length(e2 - e1))));
        // Deformation gradient, its principal stretches, and the second Piola-Kirchhoff
        // stress from them.
        let f0 = e1 * m.x + e2 * m.z;
        let f1 = e1 * m.y + e2 * m.w;
        let c00 = dot(f0, f0);
        let c01 = dot(f0, f1);
        let c11 = dot(f1, f1);
        let mid = 0.5 * (c00 + c11);
        let dev = sqrt(0.25 * (c00 - c11) * (c00 - c11) + c01 * c01);
        let l = sqrt(vec2f(mid + dev, max(mid - dev, 1e-12)));
        // Latex breaks at a stretch. The mesh is fine enough (about 0.5 mm unstretched) that
        // a hole or a crack's tip concentrates the stretch over several triangles. All three
        // vertices see the same stretch, so they agree on when.
        if (l.x >= U.rubber2.y) {
            alive[t] = 0u;
            continue;
        }
        let sh = sheet(l.x, l.y);
        let sig = sh.sig;
        var e = vec2f(1.0, 0.0);
        if (dev > 1e-6 * mid) {
            e = normalize(select(vec2f(c01, mid + dev - c00), vec2f(mid + dev - c11, c01), c00 >= c11));
        }
        let s = sig / (l * l);
        let ds = s.x - s.y;
        let s00 = s.y + ds * e.x * e.x;
        let s01 = ds * e.x * e.y;
        let s11 = s.y + ds * e.y * e.y;
        let p0 = f0 * s00 + f1 * s01;
        let p1 = f0 * s01 + f1 * s11;
        let vol = tri.rest_area * U.rubber.w;
        let h1 = -vol * (p0 * m.x + p1 * m.y);
        let h2 = -vol * (p0 * m.z + p1 * m.w);
        var grad = vec2f(m.x, m.y);
        if (i == tri.v0) {
            o.f -= h1 + h2;
            grad = -vec2f(m.x + m.z, m.y + m.w);
        } else if (i == tri.v1) {
            o.f += h1;
        } else {
            o.f += h2;
            grad = vec2f(m.z, m.w);
        }
        let cr = cross(e1, e2);
        o.n += cr;
        o.area += length(cr) / 6.0;
        rest += tri.rest_area / 3.0;
        o.stiff += vol * dot(grad, grad) * (sh.tangent + s.x);
        o.tension = max(o.tension, sig.x * U.rubber.w / (l.x * l.y));
    }
    // Stretching keeps its volume.
    if (o.area > 0.0) { o.thick = U.rubber.w * rest / o.area; }
    return o;
}

fn on_water(i: u32) -> bool {
    return mem_mass[i] > 0.0 && mem_water[i].w >= CONTACT;
}

// Storage index of the cell at `c`, or NONE where its block isn't allocated.
fn cell_at(c: vec3i) -> u32 {
    if (any(c < vec3i(0)) || any(c >= vec3i(U.dims.xyz))) { return NONE; }
    let b = atomicLoad(&block_ptr[block_slot(c >> vec3u(2u))]);
    if (b >= PENDING) { return NONE; }
    let l = vec3u(c & vec3i(3));
    return b * 64u + l.x * 16u + l.y * 4u + l.z;
}

// Blocks around rubber off the water, so the air there is simulated too.
@compute @workgroup_size(256)
fn mark_rubber(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= U.mem.x || mem_push[i].y <= 0.0 || on_water(i)) { return; }
    let c = own_cell(mem_x[i].xyz);
    let bmax = vec3i(U.bdims.xyz) - 1;
    let lo = clamp((c - 2) >> vec3u(2u), vec3i(0), bmax);
    let hi = clamp((c + 3) >> vec3u(2u), vec3i(0), bmax);
    for (var x = lo.x; x <= hi.x; x++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var z = lo.z; z <= hi.z; z++) {
                claim(vec3i(x, y, z));
            }
        }
    }
}

// Hands the rubber's pull on the water to the cells around each vertex on it, with the same
// weights the particles use. For the air, every cell with a vertex in it is a wall that moves
// with the rubber: vertices are closer than a cell, so the walls of a sheet join up.
@compute @workgroup_size(256)
fn mem_splat(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= U.mem.x || mem_mass[i] == 0.0) { return; }
    let pa = mem_push[i];
    if (pa.y <= 0.0) { return; }
    let own = cell_at(own_cell(mem_x[i].xyz));
    if (own != NONE) {
        let a = pa.y * 1e6;
        atomicAdd(&acc[acc_i(A_MW, own)], to_fp(a, FP_A));
        for (var d = 0u; d < 3u; d++) {
            atomicAdd(&acc[acc_i(A_MV + d, own)], to_fp(a * mem_v[i][d], FP_A));
        }
    }
    if (!on_water(i)) { return; }
    let x = (mem_x[i].xyz - U.origin.xyz) / h() - 0.5;
    let base = vec3i(floor(x - 0.5));
    let fx = x - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let cell = cell_at(base + vec3i(vec3u(a, b, c)));
                if (cell == NONE) { continue; }
                let w = wx[a] * wy[b] * wz[c];
                atomicAdd(&acc[acc_i(A_MF, cell)], to_fp(w * pa.x, FP_F));
                atomicAdd(&acc[acc_i(A_MA, cell)], to_fp(w * pa.y * 1e6, FP_A));
            }
        }
    }
}

// Quadratic B-spline sum around `x` (in cells, on the lattice) of cell field `fu` weighted by
// transfer field `fw`, and of the weights.
fn sample_grid(x: vec3f, fu: u32, fw: u32) -> vec2f {
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
                let w = wx[a] * wy[b] * wz[c] * from_fp(atomicLoad(&acc[acc_i(fw, cell)]), FP_W);
                s += vec2f(w * cf[cf_i(fu, cell)], w);
            }
        }
    }
    return s;
}

// The water at each vertex once the step's pressure has acted: the velocity of the faces
// around it, weighted by how much water each carries, and how full the cells around it are.
@compute @workgroup_size(256)
fn mem_sample(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= U.mem.x || mem_mass[i] == 0.0) { return; }
    let x = (mem_x[i].xyz - U.origin.xyz) / h();
    var v = vec3f(0.0);
    for (var d = 0u; d < 3u; d++) {
        let s = sample_grid(x - lattice_offset(d), F_U + d, a_wt(d));
        v[d] = select(0.0, s.x / s.y, s.y > 1e-6);
    }
    // A bulk cell holds 8 particles, each with weights summing to 1.
    let fill = sample_grid(x - 0.5, F_TYPE, A_WC).y / 8.0;
    mem_water[i] = vec4f(v, fill);
}

// The water layer the rubber drags along the surface. Water next to the rubber moves with
// it, and a change in their relative speed spreads into the water by viscosity, so the drag
// is sqrt(rho mu / pi) times the history of the change weighted by 1 / sqrt(age) (the Stokes
// layer). The weighting is a sum of decaying exponentials fitted to 1 / sqrt(pi t) from
// 10 us to 1 s, to 10%: the layer only reaches a cell's width after seconds.
// Two groups of three, as vectors: a constant array indexed in a loop would go to scratch
// memory.
const LAYER_MODES: u32 = 6u;
const LAYER_TAU = array<vec3f, 2>(vec3f(2.0e-6, 3.170e-5, 5.024e-4), vec3f(7.962e-3, 1.262e-1, 2.0));
const LAYER_W = array<vec3f, 2>(vec3f(573.42, 158.41, 38.750), vec3f(10.049, 2.2387, 0.9494));
const MU_WATER: f32 = 1.0e-3;

var<workgroup> red_min: array<vec2f, 256>;
var<workgroup> ticket: u32;

fn workgroup_min(li: u32, v: vec2f) -> vec2f {
    red_min[li] = v;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (li < off) { red_min[li] = min(red_min[li], red_min[li + off]); }
        workgroupBarrier();
    }
    return red_min[0];
}

// One substep of the rubber. Also measures the longest stable substep, and the longest
// step the water can take against the rubber's pull: like surface tension, the pull acts
// on the water from the last step's shape, which the water outruns if the step is longer
// than a capillary wave takes to cross a cell.
@compute @workgroup_size(256)
fn mem_substep(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
    // Which substep of the step this is: every workgroup of one substep draws a ticket.
    if (li == 0u) { ticket = atomicAdd(&st.mem_ticket, 1u); }
    let k = workgroupUniformLoad(&ticket) / (nwg.x * nwg.y);
    let i = flat_index(gid, nwg);
    var limits = vec2f(3.0e38);
    if (i < U.mem.x) {
        limits = substep_one(i, k);
    }
    let m = workgroup_min(li, limits);
    if (li == 0u) {
        atomicMin(&st.mem_dt, bitcast<u32>(m.x));
        atomicMin(&st.cap_dt, bitcast<u32>(m.y));
    }
}

// Returns the vertex's stable substep and capillary step.
fn substep_one(i: u32, k: u32) -> vec2f {
    let m = mem_mass[i];
    let x = mem_x[i];
    if (m == 0.0) {
        mem_x_out[i] = x;
        mem_v_out[i] = vec4f(0.0);
        return vec2f(3.0e38);
    }
    let sub = dt() / f32(max(U.mem.y, 1u));
    let t_sub = now() - dt() + f32(k + 1u) * sub;
    let vf = vertex_force(i);
    var limits = vec2f(sqrt(m / max(vf.stiff, 1e-30)), 3.0e38);
    if (vf.tension > 0.0) {
        let hh = h();
        limits.y = sqrt(U.fluid.x * hh * hh * hh / (4.0 * PI * vf.tension));
    }
    let f = vf.f + m * U.gravity.xyz;
    var v = mem_v[i].xyz;
    let nl = length(vf.n);
    let first = i * LAYER_MODES;
    // Rubber a bullet holds pushes on the bullet, not the water.
    var held = mem_v[i].w > 0.0;
    let push = select(0.0, -dot(f, vf.n / max(nl, 1e-30)), on_water(i) && nl > 0.0);
    mem_push[i] = vec2f(select(push, 0.0, held), vf.area);
    // A substep of no length only measures.
    if (sub <= 0.0) {
        mem_x_out[i] = x;
        mem_v_out[i] = mem_v[i];
        return limits;
    }
    if (on_water(i) && nl > 0.0) {
        let n = vf.n / nl;
        // Across the surface it goes with the water. Along it, the speed relative to the
        // water comes from the pull along the surface less the layer's drag, implicitly.
        let vw = mem_water[i].xyz;
        let rel = v - vw;
        let rel_t = rel - dot(rel, n) * n;
        let ft = f - dot(f, n) * n;
        let k = sqrt(U.fluid.x * MU_WATER) * vf.area;
        // Each mode decays by e over the substep and takes up g of the change in speed.
        let e = array<vec3f, 2>(exp(-sub / LAYER_TAU[0]), exp(-sub / LAYER_TAU[1]));
        let g = array<vec3f, 2>((1.0 - e[0]) * LAYER_TAU[0] / sub, (1.0 - e[1]) * LAYER_TAU[1] / sub);
        var lhs = m / sub + k * (dot(LAYER_W[0], g[0]) + dot(LAYER_W[1], g[1]));
        var rhs = m / sub * rel_t + ft;
        var z: array<vec3f, 6>;
        for (var j = 0u; j < 6u; j++) {
            z[j] = mem_layer[first + j].xyz;
            z[j] -= dot(z[j], n) * n;
        }
        for (var a = 0u; a < 2u; a++) {
            for (var b = 0u; b < 3u; b++) {
                rhs -= k * LAYER_W[a][b] * (e[a][b] * z[3u * a + b] - g[a][b] * rel_t);
            }
        }
        let rel_new = rhs / lhs;
        for (var a = 0u; a < 2u; a++) {
            for (var b = 0u; b < 3u; b++) {
                mem_layer[first + 3u * a + b] = vec4f(e[a][b] * z[3u * a + b] + g[a][b] * (rel_new - rel_t), 0.0);
            }
        }
        v = vw + rel_new;
        // Held on a bullet, it goes with the bullet across the surface instead.
        if (held) { v += dot(rel, n) * n; }
    } else if (on_water(i)) {
        // A crumb torn free of the sheet, far lighter than the water it sits in.
        v = mem_water[i].xyz;
    } else {
        for (var j = 0u; j < LAYER_MODES; j++) {
            mem_layer[first + j] = vec4f(0.0);
        }
        // Off the water, the air on either side pushes on it.
        var fa = vec3f(0.0);
        if (nl > 0.0) {
            let n = vf.n / nl;
            fa = (gas_pressure_at(x.xyz - h() * n) - gas_pressure_at(x.xyz + h() * n)) * vf.area * n;
        }
        v += sub * (f + fa) / m;
    }
    var xx = x.xyz + sub * v;
    let wet = on_water(i);
    held = held && wet;
    limits.x = min(limits.x, rubber_contact(i, &xx, &v, m, f, vf, wet, &held, t_sub, sub));
    mem_push[i].x = select(push, 0.0, held);
    if (xx.y < 0.0) {
        // The ground: the rubber can't go in, and friction holds it against sliding.
        let j = max(-v.y, 0.0);
        xx.y = 0.0;
        v.y = max(v.y, 0.0);
        let s = length(v.xz);
        let keep = select(0.0, max(0.0, 1.0 - U.rubber2.x * j / s), s > 1e-9);
        v = vec3f(v.x * keep, v.y, v.z * keep);
    }
    mem_x_out[i] = vec4f(xx, x.w);
    mem_v_out[i] = vec4f(v, select(0.0, 1.0, held));
    return limits;
}
