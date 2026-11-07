// The air: an ideal gas in every cell of the grid that isn't water, solid or behind the
// rubber, stepped explicitly in substeps of the water's step (HLL fluxes, first order).
//
// Water meets it as a wall moving with the water's face velocity, and the water in turn sees
// the gas's pressure at its surface. Where the water tears open, the new space holds only
// vapour until air flows in from a neighbouring cell, so a cavity sealed inside the water is
// at the vapour pressure, and one that opens to the air fills as fast as the air can get in.
// Past the allocated blocks the air is at rest at atmospheric pressure.
//
// Gravity is left out: the air's own weight changes its pressure by about 1 Pa over the
// balloon, and the still air past the blocks would need the same profile.

// The state of each cell, field-major: density, momentum, total energy per volume. Read
// from air_x and written to air_out; the two swap every substep like the rubber's buffers.
// Cells of water or solid hold a density of 0, so the next step can tell which gas cells are
// new.
@group(0) @binding(27) var<storage, read_write> air_x: array<f32>;
@group(0) @binding(28) var<storage, read_write> air_out: array<f32>;
// 1 for a block whose air is doing anything, which keeps it allocated in the next step.
@group(0) @binding(31) var<storage, read_write> block_active: array<u32>;

const AIR_FIELDS: u32 = 5u;
const GAMMA: f32 = 1.4;
const P_ATM: f32 = 101325.0;
// Water at 20 C, and the density air would have at that pressure: a cell that is only
// vapour is modelled as air that thin, which has the same sound speed.
const P_VAPOUR: f32 = 2339.0;
const RHO_ATM: f32 = 1.204;
const RHO_VAPOUR: f32 = 0.0278;
// The lowest gauge pressure anything can have: water boils below it.
const P_FLOOR: f32 = P_VAPOUR - P_ATM;
const CFL_AIR: f32 = 0.9;

struct Gas {
    rho: f32,
    m: vec3f,
    e: f32,
}

fn air_i(field: u32, cell: u32) -> u32 {
    return field * cap_cells() + cell;
}

fn read_gas(cell: u32) -> Gas {
    return Gas(air_x[air_i(0u, cell)], vec3f(air_x[air_i(1u, cell)], air_x[air_i(2u, cell)], air_x[air_i(3u, cell)]), air_x[air_i(4u, cell)]);
}

fn write_gas(cell: u32, g: Gas) {
    air_out[air_i(0u, cell)] = g.rho;
    air_out[air_i(1u, cell)] = g.m.x;
    air_out[air_i(2u, cell)] = g.m.y;
    air_out[air_i(3u, cell)] = g.m.z;
    air_out[air_i(4u, cell)] = g.e;
}

fn gas_at_rest(rho: f32, p: f32) -> Gas {
    return Gas(rho, vec3f(0.0), p / (GAMMA - 1.0));
}

fn ambient() -> Gas {
    return gas_at_rest(RHO_ATM, P_ATM);
}

fn no_gas() -> Gas {
    return Gas(0.0, vec3f(0.0), 0.0);
}

fn gas_p(g: Gas) -> f32 {
    return (GAMMA - 1.0) * (g.e - 0.5 * dot(g.m, g.m) / g.rho);
}

// Water evaporates into any gas thinner than its vapour, and nothing is below that pressure.
fn floored(g: Gas) -> Gas {
    var o = g;
    o.rho = max(o.rho, RHO_VAPOUR);
    o.e = max(o.e, P_VAPOUR / (GAMMA - 1.0) + 0.5 * dot(o.m, o.m) / o.rho);
    return o;
}

// Air takes part here; water, solid and cells behind the rubber are walls to it.
fn free_gas(cell: u32) -> bool {
    return cell == NONE || cf[cf_i(F_TYPE, cell)] == GAS;
}

// Brings the air over from the blocks of the last step. A block that wasn't allocated then
// was still air.
@compute @workgroup_size(64)
fn air_carry(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let b = wid.x;
    let cell = b * 64u + li;
    let s = block_list[b];
    let o = prev_ptr[s];
    var g = ambient();
    if (o < st.n_blocks_prev && block_list_prev[o] == s) {
        g = read_gas(o * 64u + li);
    }
    write_gas(cell, g);
    if (li == 0u) { block_active[b] = 0u; }
}

// Gives cells that just stopped being water the gas next to them, or vapour where there is
// none, and hands the gas's pressure to the water's solve.
@compute @workgroup_size(64)
fn air_fill(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let ty = cf[cf_i(F_TYPE, cell)];
    if (ty == WATER || ty == SOLID) {
        write_gas(cell, no_gas());
        return;
    }
    var g = read_gas(cell);
    if (g.rho <= 0.0) {
        var sum = no_gas();
        var n = 0.0;
        for (var k = 0; k < 6; k++) {
            var e = vec3i(0);
            e[k / 2] = select(-1, 1, k % 2 == 1);
            let nb = neighbour(cell, e);
            if (nb == NONE) { continue; }
            let q = read_gas(nb);
            if (q.rho > 0.0) {
                sum = Gas(sum.rho + q.rho, sum.m + q.m, sum.e + q.e);
                n += 1.0;
            }
        }
        if (n > 0.0) {
            g = Gas(sum.rho / n, sum.m / n, sum.e / n);
        } else {
            g = gas_at_rest(RHO_VAPOUR, P_VAPOUR);
        }
    }
    write_gas(cell, g);
    if (ty == GAS) { cf[cf_i(F_P, cell)] = gas_p(g) - P_ATM; }
    if (ty == RUBBER) {
        // The rubber runs through the cell with gas on both sides. Any water here is on the
        // side of the thinner gas: a cavity inside the balloon, not the air outside it.
        var p = 3.0e38;
        for (var k = 0; k < 6; k++) {
            var e = vec3i(0);
            e[k / 2] = select(-1, 1, k % 2 == 1);
            let nb = neighbour(cell, e);
            if (nb == NONE || !free_gas(nb)) { continue; }
            let q = read_gas(nb);
            if (q.rho > 0.0) { p = min(p, gas_p(q)); }
        }
        cf[cf_i(F_P, cell)] = select(gas_p(g), p, p < 3.0e38) - P_ATM;
    }
}

struct Flux {
    rho: f32,
    m: vec3f,
    e: f32,
}

fn physical_flux(g: Gas, p: f32, u: f32, d: u32) -> Flux {
    var m = g.m * u;
    m[d] += p;
    return Flux(g.rho * u, m, (g.e + p) * u);
}

// HLL flux across a face along `d` from gas `l` on the low side to `r` on the high side.
fn hll(l: Gas, r: Gas, d: u32) -> Flux {
    let pl = gas_p(l);
    let pr = gas_p(r);
    let ul = l.m[d] / l.rho;
    let ur = r.m[d] / r.rho;
    let cl = sqrt(GAMMA * pl / l.rho);
    let cr = sqrt(GAMMA * pr / r.rho);
    let sl = min(ul - cl, ur - cr);
    let sr = max(ul + cl, ur + cr);
    let fl = physical_flux(l, pl, ul, d);
    let fr = physical_flux(r, pr, ur, d);
    if (sl >= 0.0) { return fl; }
    if (sr <= 0.0) { return fr; }
    let k = 1.0 / (sr - sl);
    let a = sr * k;
    let b = -sl * k;
    let c = sl * sr * k;
    return Flux(
        a * fl.rho + b * fr.rho + c * (r.rho - l.rho),
        a * fl.m + b * fr.m + c * (r.m - l.m),
        a * fl.e + b * fr.e + c * (r.e - l.e),
    );
}

// What the gas in `own` sees across a wall moving at `w` along `d`: itself, mirrored.
fn wall_ghost(own: Gas, w: f32, d: u32) -> Gas {
    var g = own;
    let u = own.m[d] / own.rho;
    g.m[d] = own.rho * (2.0 * w - u);
    g.e = own.e + 0.5 * (g.m[d] * g.m[d] - own.m[d] * own.m[d]) / own.rho;
    return g;
}

// The gas across the face along `d` from `cell`, on the high side if `hi`, and the velocity
// of that face.
fn across(cell: u32, own: Gas, d: u32, hi: bool) -> Gas {
    var e = vec3i(0);
    e[d] = select(-1, 1, hi);
    let nb = neighbour(cell, e);
    if (nb == NONE) { return ambient(); }
    if (free_gas(nb)) { return read_gas(nb); }
    var w = cf[cf_i(F_U + d, select(cell, nb, hi))];
    // Rubber off the water moves by itself, not with the water.
    if (cf[cf_i(F_TYPE, nb)] == RUBBER) {
        w = from_fp(atomicLoad(&acc[acc_i(A_MV + d, nb)]), FP_A) / from_fp(atomicLoad(&acc[acc_i(A_MW, nb)]), FP_A);
    }
    return wall_ghost(own, w, d);
}

// Absolute pressure of the gas around `x`, from the cells that are free gas. Past the blocks
// it is still air.
fn gas_pressure_at(x: vec3f) -> f32 {
    let base = vec3i(floor((x - U.origin.xyz) / h() - 1.0));
    let fx = (x - U.origin.xyz) / h() - 0.5 - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    var s = vec2f(0.0);
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let cell = cell_at(base + vec3i(vec3u(a, b, c)));
                let w = wx[a] * wy[b] * wz[c];
                if (cell == NONE) {
                    s += vec2f(w * P_ATM, w);
                } else if (free_gas(cell)) {
                    s += vec2f(w * gas_p(read_gas(cell)), w);
                }
            }
        }
    }
    return select(P_ATM, s.x / s.y, s.y > 1e-6);
}

var<workgroup> air_red: array<f32, 64>;

@compute @workgroup_size(64)
fn air_substep(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let g = read_gas(cell);
    var limit = 3.0e38;
    if (!free_gas(cell)) {
        write_gas(cell, g);
    } else {
        var f = Flux(0.0, vec3f(0.0), 0.0);
        for (var d = 0u; d < 3u; d++) {
            let lo = hll(across(cell, g, d, false), g, d);
            let hi = hll(g, across(cell, g, d, true), d);
            f = Flux(f.rho + hi.rho - lo.rho, f.m + hi.m - lo.m, f.e + hi.e - lo.e);
        }
        let k = dt() / f32(max(U.mem.z, 1u)) / h();
        let n = floored(Gas(g.rho - k * f.rho, g.m - k * f.m, g.e - k * f.e));
        write_gas(cell, n);
        let u = n.m / n.rho;
        let p = gas_p(n);
        let c = sqrt(GAMMA * p / n.rho);
        limit = CFL_AIR * h() / (abs(u.x) + abs(u.y) + abs(u.z) + 3.0 * c);
        if (abs(p - P_ATM) > 10.0 || length(u) > 0.05) { block_active[wid.x] = 1u; }
    }
    air_red[li] = limit;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (li < off) { air_red[li] = min(air_red[li], air_red[li + off]); }
        workgroupBarrier();
    }
    if (li == 0u) { atomicMin(&st.air_dt, bitcast<u32>(air_red[0])); }
}

// Keeps the blocks whose air was doing anything in the last step, even with no water near.
@compute @workgroup_size(256)
fn keep_blocks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= st.n_blocks_prev || block_active[i] == 0u) { return; }
    claim(slot_coord(block_list_prev[i]));
}
