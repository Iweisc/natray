// Water: APIC particles on a staggered grid, with an implicit pressure solve.
//
// The pressure equation is the compressible one, with water's real sound speed:
//   p / (rho c^2 dt) - dt / rho lap(p) = p_old / (rho c^2 dt) - div(u*)
// so an impact sends a pressure wave through the water, and slow flow is incompressible.
// Pressures are gauge, relative to the ambient air.

// The step's dispatch arguments, in words.
const IND_BLOCKS: u32 = 0u;
const IND_PARTICLES: u32 = 3u;
const IND_SPRAY: u32 = 6u;
const IND_CLEAR: u32 = 9u;
const IND_ONE: u32 = 12u;
const IND_VERTS: u32 = 15u;
const IND_KEEP: u32 = 18u;
const IND_BULLETS: u32 = 21u;
// The pressure solver's.
const SOLVE_BLOCKS: u32 = 0u;
const SOLVE_ONE: u32 = 3u;

fn set_groups(slot: u32, n: u32) {
    args[slot] = min(n, 65535u);
    args[slot + 1u] = select(1u, (n + 65534u) / 65535u, n > 0u);
    args[slot + 2u] = 1u;
}

fn n_blocks() -> u32 {
    return min(atomicLoad(&st.n_blocks), U.bdims.w);
}

// Picks the step from the fastest water and the rubber, lands exactly on the stop time, and
// gates every other dispatch of the step on there being one.
@compute @workgroup_size(1)
fn begin_step() {
    var vmax = bitcast<f32>(atomicLoad(&st.max_speed));
    // A bullet's own pace counts from the step it joins in, which starts exactly when it
    // arrives.
    var arrive = U.limits.w;
    for (var k = 0u; k < U.mem.w; k++) {
        let b = bullets[k];
        if (b.info.z == 1u) {
            vmax = max(vmax, b.l.w);
        } else if (b.info.z == 0u) {
            if (b.info.x > st.t_ns) {
                arrive = min(arrive, b.info.x);
            } else {
                vmax = max(vmax, length(b.v.xyz));
            }
        }
    }
    let mem_dt = bitcast<f32>(atomicLoad(&st.mem_dt));
    let air_dt = bitcast<f32>(atomicLoad(&st.air_dt));
    var dt_s = min(U.fluid.z, bitcast<f32>(atomicLoad(&st.cap_dt)));
    if (vmax > 0.0) { dt_s = min(dt_s, U.fluid.y * h() / vmax); }
    st.dt_flow = dt_s;
    st.mem_dt_seen = mem_dt;
    st.air_dt_seen = air_dt;
    dt_s = min(dt_s, min(f32(U.mem.y) * mem_dt, f32(U.mem.z) * air_dt));
    // Growth is limited so that a step cut short to land on a stop doesn't set the pace.
    if (st.free_dt_ns > 0u) { dt_s = min(dt_s, 1.25 * f32(st.free_dt_ns) * 1e-9); }
    let free_ns = max(u32(dt_s * 1e9), 20u);
    let stop = U.limits.w;
    let live = st.t_ns < stop;
    var dt_ns = 0u;
    if (live) {
        dt_ns = min(free_ns, min(stop, arrive) - st.t_ns);
        st.free_dt_ns = free_ns;
        st.t_ns += dt_ns;
        st.step += 1u;
        // Only a step that runs rebuilds the blocks and measures the flow again. One that
        // doesn't must leave both for the next that does.
        atomicStore(&st.max_speed, 0u);
        atomicStore(&st.mem_dt, FLT_MAX_BITS);
        atomicStore(&st.cap_dt, FLT_MAX_BITS);
        atomicStore(&st.air_dt, FLT_MAX_BITS);
        atomicStore(&st.mem_ticket, 0u);
        st.n_blocks_prev = n_blocks();
        atomicStore(&st.n_blocks, 0u);
    }
    st.dt_ns = dt_ns;
    set_groups(IND_PARTICLES, select(0u, (U.dims.w + 255u) / 256u, live));
    let sprays = min(atomicLoad(&st.spray_count), U.limits.y);
    set_groups(IND_SPRAY, select(0u, (sprays + 255u) / 256u, live));
    set_groups(IND_CLEAR, select(0u, (st.n_blocks_prev * 64u + 255u) / 256u, live));
    set_groups(IND_ONE, select(0u, 1u, live));
    set_groups(IND_VERTS, select(0u, (U.mem.x + 255u) / 256u, live));
    set_groups(IND_KEEP, select(0u, (st.n_blocks_prev + 255u) / 256u, live));
    set_groups(IND_BULLETS, select(0u, U.mem.w, live));
}

// Frees the last step's blocks and zeroes their cells' transfer sums, so every cell past the
// blocks in use is always clear.
@compute @workgroup_size(256)
fn clear_blocks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= st.n_blocks_prev * 64u) { return; }
    if (i % 64u == 0u) {
        let s = block_list[i / 64u];
        atomicStore(&block_ptr[s], NONE);
        atomicStore(&slot_count[s], 0u);
        block_list_prev[i / 64u] = s;
        prev_ptr[s] = i / 64u;
    }
    for (var f = 0u; f < ACC_FIELDS; f++) {
        atomicStore(&acc[acc_i(f, i)], 0);
    }
}

fn claim(b: vec3i) {
    let s = block_slot(b);
    loop {
        if (atomicLoad(&block_ptr[s]) != NONE) { return; }
        let r = atomicCompareExchangeWeak(&block_ptr[s], NONE, PENDING);
        if (r.exchanged) { break; }
        if (r.old_value != NONE) { return; }
    }
    let idx = atomicAdd(&st.n_blocks, 1u);
    if (idx >= U.bdims.w) {
        atomicStore(&st.overflow, 1u);
        atomicStore(&block_ptr[s], NONE);
        return;
    }
    block_list[idx] = s;
    atomicStore(&block_ptr[s], idx);
}

fn own_cell(p: vec3f) -> vec3i {
    return vec3i(floor((p - U.origin.xyz) / h()));
}

// Claims every block the particle's transfer stencils touch, cells floor(x) - 1 to
// floor(x) + 2, and takes a place among the particles in its own block.
@compute @workgroup_size(256)
fn mark_blocks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= U.dims.w || P[i].vel.w < 0.5) { return; }
    let c = own_cell(P[i].pos.xyz);
    let bmax = vec3i(U.bdims.xyz) - 1;
    let lo = clamp((c - 1) >> vec3u(2u), vec3i(0), bmax);
    let hi = clamp((c + 2) >> vec3u(2u), vec3i(0), bmax);
    for (var x = lo.x; x <= hi.x; x++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var z = lo.z; z <= hi.z; z++) {
                claim(vec3i(x, y, z));
            }
        }
    }
    let rank = atomicAdd(&slot_count[block_slot(c >> vec3u(2u))], 1u);
    P[i].c2.w = bitcast<f32>(rank);
}

@compute @workgroup_size(1)
fn blocks_ready() {
    set_groups(IND_BLOCKS, select(0u, n_blocks(), st.dt_ns > 0u));
}

var<workgroup> scan_part: array<u32, 256>;

// Where each block's particles start in `sorted`: an exclusive prefix sum of the counts in
// one workgroup. Each thread sums a run of blocks, the run totals are scanned, then each
// thread writes its run.
@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) li: u32) {
    let n = n_blocks();
    let chunk = (n + 255u) / 256u;
    let lo = min(li * chunk, n);
    let hi = min(lo + chunk, n);
    var sum = 0u;
    for (var b = lo; b < hi; b++) {
        sum += atomicLoad(&slot_count[block_list[b]]);
    }
    scan_part[li] = sum;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off *= 2u) {
        var v = scan_part[li];
        if (li >= off) { v += scan_part[li - off]; }
        workgroupBarrier();
        scan_part[li] = v;
        workgroupBarrier();
    }
    var run = scan_part[li] - sum;
    for (var b = lo; b < hi; b++) {
        block_start[b] = run;
        run += atomicLoad(&slot_count[block_list[b]]);
    }
    if (li == 255u) { block_start[n] = scan_part[255]; }
}

@compute @workgroup_size(256)
fn sort_particles(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= U.dims.w || P[i].vel.w < 0.5) { return; }
    let b = atomicLoad(&block_ptr[block_slot(own_cell(P[i].pos.xyz) >> vec3u(2u))]);
    if (b >= PENDING) { return; }
    sorted[block_start[b] + bitcast<u32>(P[i].c2.w)] = i;
}

// The particles of a block reach cells from one below it to two above, a 7x7x7 tile.
const TILE: u32 = 343u;

var<workgroup> links: array<u32, 27>;
var<workgroup> n_here: u32;

fn tile_index(c: vec3i, corner: vec3i) -> u32 {
    let t = vec3u(c - corner);
    return t.x * 49u + t.y * 7u + t.z;
}

// Storage index of tile cell `t`, or NONE.
fn tile_cell(t: u32) -> u32 {
    let rel = vec3i(vec3u(t / 49u, (t / 7u) % 7u, t % 7u)) - 1;
    let nb = (rel + 4) / 4 - 1;
    let b = links[u32((nb.x + 1) * 9 + (nb.y + 1) * 3 + nb.z + 1)];
    if (b == NONE) { return NONE; }
    let l = vec3u(rel - nb * 4);
    return b * 64u + l.x * 16u + l.y * 4u + l.z;
}

// Quadratic B-spline weights of the three nodes along one axis, for a particle `fx` cells
// past the first.
fn axis_weights(fx: f32) -> vec3f {
    return vec3f(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}

// Sums in the tile go to the matching transfer field, see a_mom and a_wt.
var<workgroup> tile: array<atomic<i32>, 2744>;
var<workgroup> own: array<atomic<u32>, 64>;

// Adds a particle at `x` (in cells) to the lattice offset `o` in a cell: `val` extended by
// `grad` to each node into field `fa`, and the weight itself into field `fw`. Weights are
// kept per axis so nothing here is indexed by a loop counter, which would put it in scratch
// memory.
fn splat(x: vec3f, o: vec3f, corner: vec3i, fa: u32, val: f32, grad: vec3f, fw: u32, scale: f32) {
    let base = vec3i(floor(x - o - 0.5));
    let fx = x - o - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    let t0 = tile_index(base, corner);
    let hg = grad * h();
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let weight = wx[a] * wy[b] * wz[c];
                let t = t0 + a * 49u + b * 7u + c;
                let node = val + dot(hg, vec3f(f32(a), f32(b), f32(c)) - fx);
                atomicAdd(&tile[fa * TILE + t], to_fp(weight * node, scale));
                atomicAdd(&tile[fw * TILE + t], to_fp(weight, FP_W));
            }
        }
    }
}

// Particle to grid, one block per workgroup. The block's particles add into a tile in
// shared memory, and only the tile's sums go out to the grid. It also records the block's
// neighbours for the kernels after it.
@compute @workgroup_size(256)
fn p2g(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let b = wid.x;
    let bc = slot_coord(block_list[b]);
    if (li < 27u) {
        let nb = bc + vec3i(vec3u(li / 9u, (li / 3u) % 3u, li % 3u)) - 1;
        var idx = NONE;
        if (all(nb >= vec3i(0)) && all(nb < vec3i(U.bdims.xyz))) {
            idx = atomicLoad(&block_ptr[block_slot(nb)]);
            if (idx >= PENDING) { idx = NONE; }
        }
        links[li] = idx;
        block_nb[b * 27u + li] = idx;
    }
    if (li == 0u) { n_here = block_start[b + 1u] - block_start[b]; }
    if (li < 64u) { atomicStore(&own[li], 0u); }
    for (var t = li; t < 8u * TILE; t += 256u) {
        atomicStore(&tile[t], 0);
    }
    let count = workgroupUniformLoad(&n_here);
    if (count == 0u) {
        if (li < 64u) { atomicStore(&acc[acc_i(A_N, b * 64u + li)], 0); }
        return;
    }
    let corner = bc * 4 - 1;
    let first = block_start[b];
    for (var k = li; k < count; k += 256u) {
        let p = P[sorted[first + k]];
        let x = (p.pos.xyz - U.origin.xyz) / h();
        splat(x, vec3f(0.0, 0.5, 0.5), corner, a_mom(0u), p.vel.x, p.c0.xyz, a_wt(0u), FP_M);
        splat(x, vec3f(0.5, 0.0, 0.5), corner, a_mom(1u), p.vel.y, p.c1.xyz, a_wt(1u), FP_M);
        splat(x, vec3f(0.5, 0.5, 0.0), corner, a_mom(2u), p.vel.z, p.c2.xyz, a_wt(2u), FP_M);
        splat(x, vec3f(0.5), corner, A_PC, p.c0.w * 1e-6, vec3f(0.0), A_WC, FP_P);
        let l = vec3u(own_cell(p.pos.xyz) - bc * 4);
        atomicAdd(&own[l.x * 16u + l.y * 4u + l.z], 1u);
    }
    workgroupBarrier();
    for (var t = li; t < TILE; t += 256u) {
        let cell = tile_cell(t);
        if (cell == NONE) { continue; }
        for (var f = 0u; f < 8u; f++) {
            let v = atomicLoad(&tile[f * TILE + t]);
            if (v != 0) { atomicAdd(&acc[acc_i(f, cell)], v); }
        }
    }
    if (li < 64u) { atomicStore(&acc[acc_i(A_N, b * 64u + li)], i32(atomicLoad(&own[li]))); }
}

fn is_ground(c: vec3i) -> bool {
    return U.origin.y + (f32(c.y) + 0.5) * h() < 0.0;
}

// The knot closes the balloon: a solid plug above the tie, a little wider than the neck.
fn in_knot(c: vec3i) -> bool {
    let p = U.origin.xyz + (vec3f(c) + 0.5) * h() - U.knot.xyz;
    let r = U.knot.w + 0.5 * h();
    return p.y > 0.0 && p.y < 2.0 * r && length(p.xz) < r;
}

// Every per-cell kernel runs one workgroup per block. The block capacity stays within one
// dispatch dimension, so the block is the workgroup's x.

// Face velocities from the transferred momentum, gravity, and what kind of cell this is.
@compute @workgroup_size(64)
fn grid_prep(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    var ty = GAS;
    let c = cell_coord(cell);
    if (is_ground(c) || in_knot(c)) {
        ty = SOLID;
    } else if (atomicLoad(&acc[acc_i(A_N, cell)]) > 0) {
        ty = WATER;
    }
    cf[cf_i(F_TYPE, cell)] = ty;
    for (var d = 0u; d < 3u; d++) {
        let w = from_fp(atomicLoad(&acc[acc_i(a_wt(d), cell)]), FP_W);
        var u = 0.0;
        if (w > 1e-6) {
            u = from_fp(atomicLoad(&acc[acc_i(a_mom(d), cell)]), FP_M) / w + dt() * U.gravity[d];
        }
        cf[cf_i(F_U + d, cell)] = u;
    }
    let wc = from_fp(atomicLoad(&acc[acc_i(A_WC, cell)]), FP_W);
    let ma = from_fp(atomicLoad(&acc[acc_i(A_MA, cell)]), FP_A);
    if (ty == GAS && atomicLoad(&acc[acc_i(A_MW, cell)]) > 0) {
        ty = RUBBER;
        cf[cf_i(F_TYPE, cell)] = ty;
    }
    var p = 0.0;
    if (ty == WATER && wc > 1e-6) {
        p = from_fp(atomicLoad(&acc[acc_i(A_PC, cell)]), FP_P) / wc * 1e6;
    } else if (ty != WATER && ty != SOLID && ma > 0.0) {
        // Behind the rubber the water is held at the rubber's pressure. Elsewhere the gas
        // sets it, in air_fill.
        ty = COVERED;
        cf[cf_i(F_TYPE, cell)] = ty;
        p = from_fp(atomicLoad(&acc[acc_i(A_MF, cell)]), FP_F) / ma * 1e6;
    }
    cf[cf_i(F_P, cell)] = p;
}

// Water can't flow into the ground but can lift off it. A face belongs to the cell on its
// high side, `lo` is the cell on the other.
fn bounded(u: f32, hi: f32, lo: f32) -> f32 {
    if (lo == SOLID && hi == SOLID) { return 0.0; }
    if (lo == SOLID) { return max(u, 0.0); }
    if (hi == SOLID) { return min(u, 0.0); }
    return u;
}

// Face `d` of `cell`, which has `lo` on the other side of it.
fn face_u(cell: u32, lo: u32, d: u32) -> f32 {
    if (cell == NONE) { return 0.0; }
    return bounded(cf[cf_i(F_U + d, cell)], cf[cf_i(F_TYPE, cell)], cell_type(lo));
}

// Water under more tension than its vapour pressure allows boils: the solve's pressure is
// held there and the water pulls apart instead.
fn pressure_of(cell: u32) -> f32 {
    if (cell == NONE) { return 0.0; }
    let p = cf[cf_i(F_P, cell)];
    return select(p, max(p, P_FLOOR), cf[cf_i(F_TYPE, cell)] == WATER);
}

var<workgroup> zl: array<f32, 64>;
var<workgroup> wl: array<u32, 64>;

// Symmetric red-black Gauss-Seidel sweeps over the part of the system inside this block,
// starting from zero. Always the same symmetric positive definite operator, as conjugate
// gradients needs, and it runs inside the passes that are there anyway. Every thread of the
// workgroup has to call it.
const SWEEPS: u32 = 2u;

fn precondition(li: u32, r: f32, diag: f32, water: bool) -> f32 {
    zl[li] = 0.0;
    wl[li] = select(0u, 1u, water);
    workgroupBarrier();
    let l = vec3u(li / 16u, (li / 4u) % 4u, li % 4u);
    let colour = (l.x + l.y + l.z) % 2u;
    for (var k = 0u; k < 2u * SWEEPS + 1u; k++) {
        if (water && colour == k % 2u) {
            var s = r;
            if (l.x > 0u && wl[li - 16u] != 0u) { s += zl[li - 16u]; }
            if (l.x < 3u && wl[li + 16u] != 0u) { s += zl[li + 16u]; }
            if (l.y > 0u && wl[li - 4u] != 0u) { s += zl[li - 4u]; }
            if (l.y < 3u && wl[li + 4u] != 0u) { s += zl[li + 4u]; }
            if (l.z > 0u && wl[li - 1u] != 0u) { s += zl[li - 1u]; }
            if (l.z < 3u && wl[li + 1u] != 0u) { s += zl[li + 1u]; }
            zl[li] = s / diag;
        }
        workgroupBarrier();
    }
    return zl[li];
}

var<workgroup> red: array<vec4f, 64>;

fn block_sum(li: u32, v: vec4f, block: u32) {
    red[li] = v;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (li < off) { red[li] += red[li + off]; }
        workgroupBarrier();
    }
    if (li == 0u) { partial[block] = red[0]; }
}

var<workgroup> red1: array<vec4f, 256>;

fn total(li: u32) -> vec4f {
    let n = n_blocks();
    var s = vec4f(0.0);
    for (var i = li; i < n; i += 256u) {
        s += partial[i];
    }
    red1[li] = s;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (li < off) { red1[li] += red1[li + off]; }
        workgroupBarrier();
    }
    return red1[0];
}

fn converged(rr: f32) -> bool {
    return sqrt(rr) <= U.solve.x * max(st.b_norm, 1.0);
}

// Preconditioned conjugate gradients, warm started from the pressure the particles carry.
//
// The equation is scaled by rho h^2 / dt: the diagonal is the compressibility term plus one
// per open neighbour, and the right-hand side is in pascals. Air neighbours are known
// pressures and the ground is a wall.
@compute @workgroup_size(64)
fn solve_init(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let water = cf[cf_i(F_TYPE, cell)] == WATER;
    var diag = 1.0;
    var r = 0.0;
    var b = 0.0;
    if (water) {
        let hh = h();
        let cs = U.gravity.w * dt();
        let kc = hh * hh / (cs * cs);
        var div = 0.0;
        var off = 0.0;
        diag = kc;
        b = kc * cf[cf_i(F_P, cell)];
        for (var d = 0u; d < 3u; d++) {
            var e = vec3i(0);
            e[d] = 1;
            div += face_u(neighbour(cell, e), cell, d) - face_u(cell, neighbour(cell, -e), d);
            for (var s = -1; s <= 1; s += 2) {
                let nb = neighbour(cell, s * e);
                let nt = cell_type(nb);
                if (nt == SOLID) { continue; }
                diag += 1.0;
                if (nt == WATER) {
                    off -= cf[cf_i(F_P, nb)];
                } else {
                    b += pressure_of(nb);
                }
            }
        }
        // Where a bullet's cavity is opening, the water has to make way for it.
        div -= from_fp(atomicLoad(&acc[acc_i(A_SRC, cell)]), FP_Q) / (hh * hh);
        b -= U.fluid.x * hh / dt() * div;
        r = b - diag * cf[cf_i(F_P, cell)] - off;
        cf[cf_i(F_DIAG, cell)] = diag;
        cf[cf_i(F_R, cell)] = r;
    }
    let z = precondition(li, r, diag, water);
    if (water) { cf[cf_i(F_Z, cell)] = z; }
    block_sum(li, vec4f(r * z, r * r, b * b, 0.0), wid.x);
}

// The solver's own dispatch arguments turn its iterations off once it has converged.
@compute @workgroup_size(256)
fn solve_start(@builtin(local_invocation_index) li: u32) {
    let s = total(li);
    if (li != 0u) { return; }
    st.rz = s.x;
    st.beta = 0.0;
    st.b_norm = sqrt(s.z);
    st.r_norm = sqrt(s.y);
    st.solve_iters = 0u;
    let run = s.x > 0.0 && !converged(s.y);
    set_groups(SOLVE_BLOCKS, select(0u, n_blocks(), run));
    set_groups(SOLVE_ONE, select(0u, 1u, run));
}

// The new search direction at a water cell, from the one in direction field `k`.
fn direction(cell: u32, k: u32) -> f32 {
    return cf[cf_i(F_Z, cell)] + st.beta * cf[cf_i(F_D + k, cell)];
}

@compute @workgroup_size(64)
fn solve_q(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    var v = vec4f(0.0);
    if (cf[cf_i(F_TYPE, cell)] == WATER) {
        let old = st.solve_iters % 2u;
        let dn = direction(cell, old);
        var q = cf[cf_i(F_DIAG, cell)] * dn;
        for (var d = 0u; d < 3u; d++) {
            var e = vec3i(0);
            e[d] = 1;
            for (var k = -1; k <= 1; k += 2) {
                let nb = neighbour(cell, k * e);
                if (cell_type(nb) == WATER) { q -= direction(nb, old); }
            }
        }
        cf[cf_i(F_D + 1u - old, cell)] = dn;
        cf[cf_i(F_Q, cell)] = q;
        v.x = dn * q;
    }
    block_sum(li, v, wid.x);
}

@compute @workgroup_size(256)
fn solve_alpha(@builtin(local_invocation_index) li: u32) {
    let s = total(li);
    if (li != 0u) { return; }
    st.alpha = select(0.0, st.rz / s.x, s.x > 0.0);
}

@compute @workgroup_size(64)
fn solve_update(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let water = cf[cf_i(F_TYPE, cell)] == WATER;
    var diag = 1.0;
    var r = 0.0;
    if (water) {
        let a = st.alpha;
        cf[cf_i(F_P, cell)] += a * cf[cf_i(F_D + 1u - st.solve_iters % 2u, cell)];
        r = cf[cf_i(F_R, cell)] - a * cf[cf_i(F_Q, cell)];
        cf[cf_i(F_R, cell)] = r;
        diag = cf[cf_i(F_DIAG, cell)];
    }
    let z = precondition(li, r, diag, water);
    if (water) { cf[cf_i(F_Z, cell)] = z; }
    block_sum(li, vec4f(r * z, r * r, 0.0, 0.0), wid.x);
}

// Runs on the step's arguments, not the solver's, since it is the one that switches the
// solver off. After that it still runs once per iteration, and does nothing.
@compute @workgroup_size(256)
fn solve_beta(@builtin(local_invocation_index) li: u32) {
    let s = total(li);
    if (li != 0u || args[SOLVE_ONE] == 0u) { return; }
    st.solve_iters += 1u;
    st.r_norm = sqrt(s.y);
    st.beta = select(0.0, s.x / st.rz, st.rz > 0.0);
    st.rz = s.x;
    let done = converged(s.y);
    if (done || st.solve_iters >= U.limits.z) {
        st.iters_max = max(st.iters_max, st.solve_iters);
        st.unconverged += select(1u, 0u, done);
        set_groups(SOLVE_BLOCKS, 0u);
        set_groups(SOLVE_ONE, 0u);
    }
}

// Subtracts the pressure gradient from every face next to water, and holds every face to
// the ground.
@compute @workgroup_size(64)
fn project(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let ty = cf[cf_i(F_TYPE, cell)];
    let k = dt() / (U.fluid.x * h());
    for (var d = 0u; d < 3u; d++) {
        var e = vec3i(0);
        e[d] = 1;
        let nb = neighbour(cell, -e);
        let nt = cell_type(nb);
        var u = bounded(cf[cf_i(F_U + d, cell)], ty, nt);
        if (ty != SOLID && nt != SOLID && (ty == WATER || nt == WATER)) {
            u -= k * (pressure_of(cell) - pressure_of(nb));
        }
        cf[cf_i(F_U + d, cell)] = u;
    }
}

// Faces away from water have only what the particles brought and gravity: the pressure never
// reached them. Particles and rubber at the surface read faces a cell or two out, so the
// projected velocity is carried out to them, a layer of faces per pass, each face taking the
// mean of the ones already set around it. The first pass works in the solver's spare fields.
const F_EXT: u32 = 4u;
const F_EXT_SET: u32 = 7u;

fn face_set(cell: u32, d: u32) -> bool {
    if (cell == NONE) { return false; }
    var e = vec3i(0);
    e[d] = 1;
    return cf[cf_i(F_TYPE, cell)] == WATER || cell_type(neighbour(cell, -e)) == WATER;
}

@compute @workgroup_size(64)
fn extend_1(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    var done = 0u;
    for (var d = 0u; d < 3u; d++) {
        var u = cf[cf_i(F_U + d, cell)];
        if (face_set(cell, d)) {
            done |= 1u << d;
        } else {
            var sum = 0.0;
            var n = 0.0;
            for (var k = 0; k < 6; k++) {
                var e = vec3i(0);
                e[k / 2] = select(-1, 1, k % 2 == 1);
                let nb = neighbour(cell, e);
                if (face_set(nb, d)) {
                    sum += cf[cf_i(F_U + d, nb)];
                    n += 1.0;
                }
            }
            if (n > 0.0) {
                u = sum / n;
                done |= 1u << d;
            }
        }
        cf[cf_i(F_EXT + d, cell)] = u;
    }
    cf[cf_i(F_EXT_SET, cell)] = f32(done);
}

fn ext_set(cell: u32, d: u32) -> bool {
    return cell != NONE && ((u32(cf[cf_i(F_EXT_SET, cell)]) >> d) & 1u) != 0u;
}

@compute @workgroup_size(64)
fn extend_2(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x * 64u + li;
    let ty = cf[cf_i(F_TYPE, cell)];
    for (var d = 0u; d < 3u; d++) {
        var u = cf[cf_i(F_U + d, cell)];
        if (ext_set(cell, d)) {
            u = cf[cf_i(F_EXT + d, cell)];
        } else {
            var sum = 0.0;
            var n = 0.0;
            for (var k = 0; k < 6; k++) {
                var e = vec3i(0);
                e[k / 2] = select(-1, 1, k % 2 == 1);
                let nb = neighbour(cell, e);
                if (ext_set(nb, d)) {
                    sum += cf[cf_i(F_EXT + d, nb)];
                    n += 1.0;
                }
            }
            if (n > 0.0) { u = sum / n; }
        }
        var e = vec3i(0);
        e[d] = 1;
        cf[cf_i(F_U + d, cell)] = bounded(u, ty, cell_type(neighbour(cell, -e)));
    }
}

fn drag_cd(re: f32) -> f32 {
    if (re < 1e-4) { return 24.0 / 1e-4; }
    return 24.0 / re * (1.0 + 0.15 * pow(re, 0.687)) + 0.42 / (1.0 + 42500.0 * pow(re, -1.16));
}

// Drops evaporate by diffusion, d(r^2)/dt = -2 D (rho_sat - rho_vapour) / rho_water, in air
// at 20 C and 85% humidity.
const EVAPORATION: f32 = 1.3e-10;

// Drops fly through still air until they land. This runs before g2p, so drops made in this
// step start moving in the next.
@compute @workgroup_size(256)
fn spray_update(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= min(atomicLoad(&st.spray_count), U.limits.y)) { return; }
    let s = spray[i];
    var r = s.pos.w;
    if (r <= 0.0) { return; }
    let step = dt();
    let sp = length(s.vel.xyz);
    let re = 2.0 * r * sp * RHO_AIR / MU_AIR;
    let k = 3.0 * RHO_AIR * drag_cd(re) * sp / (8.0 * U.fluid.x * r);
    // Implicit, so the finest mist is stable at any step.
    let v = (s.vel.xyz + step * U.gravity.xyz) / (1.0 + k * step);
    r = sqrt(max(r * r - EVAPORATION * step, 0.0));
    // Smaller than this, a drop is gone within a few hundredths of a second.
    if (r < 2.5e-6) { r = 0.0; }
    let pos = s.pos.xyz + step * v;
    if (pos.y <= r) {
        // A parcel lands with the volume of all its drops.
        landed[i] = vec4f(pos.x, pos.z, r * pow(s.vel.w, 1.0 / 3.0), now());
        r = 0.0;
    }
    spray[i].pos = vec4f(pos, r);
    spray[i].vel = vec4f(v, s.vel.w);
}

// Largest drop that survives moving through still air at `speed`.
fn breakup_diameter(speed: f32) -> f32 {
    return WE_CRIT * SIGMA_W / (RHO_AIR * max(speed * speed, 1e-4));
}

const WE_CRIT: f32 = 12.0;
const BREAKUP_DROPS: u32 = 4u;

// A particle that leaves the resolved flow is a blob of water about one particle spacing
// across, which breaks up below the grid scale. It becomes a few drops with a spread of
// sizes, and whatever volume they don't account for becomes a parcel of finer drops that
// is rendered as a cloud. Fast blobs are shredded by the air into fine spray.
fn break_up(x: vec3f, v: vec3f, id: u32) {
    let first = atomicAdd(&st.spray_count, BREAKUP_DROPS + 1u);
    if (first >= U.limits.y) { return; }
    var seed = hash3(st.step, id, 0xb4eau);
    let hs = U.fluid.w;
    let vol = hs * hs * hs;
    let speed = length(v);
    // Ligaments at the grid scale pinch off into drops about half its size.
    let d_med = min(0.55 * hs, 0.5 * breakup_diameter(speed));
    var dd: array<f32, BREAKUP_DROPS>;
    var used = 0.0;
    for (var k = 0u; k < BREAKUP_DROPS; k++) {
        dd[k] = clamp(d_med * exp(0.5 * rnd_gauss(&seed)), 2e-5, 1.24 * hs);
        used += 0.5236 * dd[k] * dd[k] * dd[k];
    }
    let df = 0.5 * d_med;
    // Only drops too small to see one by one belong in a parcel. Slower blobs put their
    // whole volume into the visible drops.
    let fine = df < 1e-4;
    let fit = pow(vol / used, 1.0 / 3.0);
    let scale = select(fit, min(1.0, fit), fine);
    for (var k = 0u; k < BREAKUP_DROPS; k++) {
        let slot = first + k;
        if (slot >= U.limits.y) { return; }
        let dk = dd[k] * scale;
        // Fragments separate at about the capillary speed, plus a share of the airflow.
        let spread = sqrt(SIGMA_W / (U.fluid.x * dk)) + 0.05 * speed;
        let u = normalize(vec3f(rnd_gauss(&seed), rnd_gauss(&seed), rnd_gauss(&seed)) + 1e-6);
        let off = (vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5) * hs;
        spray[slot].pos = vec4f(x + off, 0.5 * dk);
        spray[slot].vel = vec4f(v + u * spread * rnd(&seed), 1.0);
        spray[slot].air = vec4f(0.0, 0.0, 0.0, now());
    }
    let slot = first + BREAKUP_DROPS;
    if (slot >= U.limits.y) { return; }
    let n = (vol - used * scale * scale * scale) / (0.5236 * df * df * df);
    // Weight 1 means a single drop, so a parcel needs at least two.
    spray[slot].pos = vec4f(x, select(0.0, 0.5 * df, fine && n >= 2.0));
    spray[slot].vel = vec4f(v, max(n, 2.0));
    spray[slot].air = vec4f(0.0, 0.0, 0.0, now());
}

// The grid around the block, in the same field order as the transfer sums: each face's
// velocity and weight, then the cell's pressure and 1 where it is water.
var<workgroup> tg: array<f32, 2744>;
var<workgroup> vmax: atomic<u32>;

struct Gathered {
    // Sum of weight times field `fa`, the same times the offset to each node (in m), and
    // the sum of weight times field `fw`.
    s: f32,
    g: vec3f,
    w: f32,
}

fn gather(x: vec3f, o: vec3f, corner: vec3i, fa: u32, fw: u32) -> Gathered {
    let base = vec3i(floor(x - o - 0.5));
    let fx = x - o - vec3f(base);
    let wx = axis_weights(fx.x);
    let wy = axis_weights(fx.y);
    let wz = axis_weights(fx.z);
    let t0 = tile_index(base, corner);
    var r = Gathered(0.0, vec3f(0.0), 0.0);
    for (var a = 0u; a < 3u; a++) {
        for (var b = 0u; b < 3u; b++) {
            for (var c = 0u; c < 3u; c++) {
                let weight = wx[a] * wy[b] * wz[c];
                let t = t0 + a * 49u + b * 7u + c;
                let u = weight * tg[fa * TILE + t];
                r.s += u;
                r.g += u * (vec3f(f32(a), f32(b), f32(c)) - fx);
                r.w += weight * tg[fw * TILE + t];
            }
        }
    }
    r.g *= h();
    return r;
}

// Moves one particle with the grid, or turns it into spray or wet ground. Returns its speed,
// or 0 once it is gone.
fn g2p_one(i: u32, corner: vec3i) -> f32 {
    let p = P[i];
    let hh = h();
    let X = (p.pos.xyz - U.origin.xyz) / hh;
    let gu = gather(X, vec3f(0.0, 0.5, 0.5), corner, a_mom(0u), a_wt(0u));
    let gv = gather(X, vec3f(0.5, 0.0, 0.5), corner, a_mom(1u), a_wt(1u));
    let gw = gather(X, vec3f(0.5, 0.5, 0.0), corner, a_mom(2u), a_wt(2u));
    let gp = gather(X, vec3f(0.5), corner, A_PC, A_WC);
    var v = vec3f(gu.s, gv.s, gw.s);
    let m_near = gu.w + gv.w + gw.w;
    let step = dt();
    var x = p.pos.xyz + step * v;
    if (U.mem.w > 0u) { carve(&x, &v, i); }
    let spacing = U.fluid.w;

    let lo = U.origin.xyz + 3.5 * hh;
    let hi = U.origin.xyz + (vec3f(U.dims.xyz) - 4.5) * hh;
    let escaped = any(x.xz < lo.xz) || any(x.xz > hi.xz) || x.y > hi.y;
    // 1 inside bulk water, about 0.3 in a sheet one particle thick, less once that sheet
    // stretches. Sheets and threads that thin break up into drops in reality, and neither
    // the grid nor the renderer can resolve them anyway.
    let fill = m_near / 24.0;
    // Jets and sheets a particle or two thick are unstable in the air: jets neck into beads
    // (Rayleigh-Plateau) and sheets tear open, within about three capillary times
    // 3 sqrt(rho r^3 / sigma) of a thread this thin. A film on the ground is held flat by it.
    let on_ground = x.y < 1.5 * spacing;
    let r_thin = 0.75 * spacing;
    let t_thin = 3.0 * sqrt(U.fluid.x * r_thin * r_thin * r_thin / SIGMA_W);
    // Under the rubber nothing is thin: the rubber holds the water in.
    let covered = atomicLoad(&acc[acc_i(A_MA, tile_cell(tile_index(vec3i(floor(X)), corner)))]) > 0;
    let thin_age = select(0.0, p.c1.w + step, fill < 0.45 && !on_ground && !covered);
    if (escaped || (!covered && fill < 0.2) || thin_age > t_thin) {
        if (on_ground) {
            // Too thin for the grid, so it soaks into the wet ground where it is.
            ground_touch(x);
        } else {
            break_up(x, v, i);
        }
        P[i].vel = vec4f(v, 0.0);
        return 0.0;
    }
    var xx = x;
    var vv = v;
    if (xx.y < 0.3 * spacing) {
        xx.y = 0.3 * spacing;
        vv.y = max(vv.y, 0.0);
    }
    if (xx.y < 0.7 * spacing) { ground_touch(xx); }
    let cs = 4.0 / (hh * hh);
    P[i].pos = vec4f(xx, p.pos.w);
    P[i].vel = vec4f(vv, 1.0);
    P[i].c0 = vec4f(gu.g * cs, select(0.0, gp.s / gp.w, gp.w > 1e-6));
    P[i].c1 = vec4f(gv.g * cs, thin_age);
    P[i].c2 = vec4f(gw.g * cs, 0.0);
    return length(vv);
}

// Grid to particle, one block per workgroup: the tile around the block comes into shared
// memory once, and every particle of the block reads from there.
@compute @workgroup_size(256)
fn g2p(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let b = wid.x;
    if (li < 27u) { links[li] = block_nb[b * 27u + li]; }
    if (li == 0u) {
        n_here = block_start[b + 1u] - block_start[b];
        atomicStore(&vmax, 0u);
    }
    let count = workgroupUniformLoad(&n_here);
    if (count == 0u) { return; }
    for (var t = li; t < TILE; t += 256u) {
        let cell = tile_cell(t);
        let water = cell != NONE && cf[cf_i(F_TYPE, cell)] == WATER;
        for (var d = 0u; d < 3u; d++) {
            var u = 0.0;
            var w = 0.0;
            if (cell != NONE) {
                u = cf[cf_i(F_U + d, cell)];
                w = from_fp(atomicLoad(&acc[acc_i(a_wt(d), cell)]), FP_W);
            }
            tg[a_mom(d) * TILE + t] = u;
            tg[a_wt(d) * TILE + t] = w;
        }
        tg[A_PC * TILE + t] = select(0.0, pressure_of(cell), water);
        tg[A_WC * TILE + t] = select(0.0, 1.0, water);
    }
    workgroupBarrier();
    let corner = slot_coord(block_list[b]) * 4 - 1;
    let first = block_start[b];
    var fastest = 0.0;
    for (var k = li; k < count; k += 256u) {
        fastest = max(fastest, g2p_one(sorted[first + k], corner));
    }
    atomicMax(&vmax, bitcast<u32>(fastest));
    workgroupBarrier();
    if (li == 0u) { atomicMax(&st.max_speed, atomicLoad(&vmax)); }
}
