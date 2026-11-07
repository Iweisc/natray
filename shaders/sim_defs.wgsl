// Shared by every simulation kernel: parameters, the GPU-driven clock, and the sparse grid.
//
// The grid covers the whole domain in 4x4x4-cell blocks, but only blocks near water are
// allocated, every step, from the particles. Cell (i, j, k) spans [i, i + 1] h from the
// domain corner. It owns the three faces on its low sides: face u at (i, j + 1/2, k + 1/2),
// v at (i + 1/2, j, k + 1/2), w at (i + 1/2, j + 1/2, k).

struct Params {
    origin: vec4f,      // domain corner, w = cell size h
    dims: vec4u,        // domain size in cells, w = particle count
    bdims: vec4u,       // domain size in blocks, w = block capacity
    gravity: vec4f,     // xyz, w = sound speed in water
    fluid: vec4f,       // x water density, y CFL number, z longest step (s), w particle spacing
    wet: vec4f,         // x wet map half size
    limits: vec4u,      // x wet map resolution, y spray capacity, z solver iterations per step, w when to stop (ns)
    solve: vec4f,       // x relative tolerance
    mem: vec4u,         // x rubber vertex count, y rubber substeps per step, z air substeps per step, w bullets
    rubber: vec4f,      // xyz Ogden moduli (Pa), w uninflated thickness
    rubber2: vec4f,     // x the rubber's friction on the ground, y the stretch it breaks at
    knot: vec4f,        // where the neck is tied, w = the neck's radius there
}

// Word 16 is the spray slot count, which the keyframe packer reads.
struct State {
    t_ns: u32,
    dt_ns: u32,
    step: u32,
    n_blocks: atomic<u32>,
    max_speed: atomic<u32>,
    n_blocks_prev: u32,
    overflow: atomic<u32>,
    solve_iters: u32,
    rz: f32,
    alpha: f32,
    b_norm: f32,
    r_norm: f32,
    // The last step before it was cut short to land on a stop time.
    free_dt_ns: u32,
    // Most solver iterations any step took, and how many steps ran out of them, since the
    // CPU last reset these.
    iters_max: u32,
    unconverged: u32,
    beta: f32,
    spray_count: atomic<u32>,
    // Longest stable rubber substep, and the longest step the water can take against the
    // rubber's pull, since the last step began (f32 bits).
    mem_dt: atomic<u32>,
    cap_dt: atomic<u32>,
    // What the last step was given: the rubber's substep, and the step it could have taken
    // had the substeps not been the limit.
    mem_dt_seen: f32,
    dt_flow: f32,
    // The same for the air's substep.
    air_dt: atomic<u32>,
    air_dt_seen: f32,
    // Drawn by each workgroup of a rubber substep, so it knows which substep it is.
    mem_ticket: atomic<u32>,
}

// c2.w holds the particle's place among those in its block while they are being sorted.
struct Particle { pos: vec4f, vel: vec4f, c0: vec4f, c1: vec4f, c2: vec4f }
struct Spray { pos: vec4f, vel: vec4f, air: vec4f }

@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read_write> st: State;
@group(0) @binding(2) var<storage, read_write> P: array<Particle>;
// Per block of the whole domain: its index among the allocated blocks, and how many
// particles are in it.
@group(0) @binding(3) var<storage, read_write> block_ptr: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> slot_count: array<atomic<u32>>;
// Per allocated block: where it is in the domain, the 27 blocks around it (itself in the
// middle), and where its particles start in `sorted`.
@group(0) @binding(5) var<storage, read_write> block_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_nb: array<u32>;
@group(0) @binding(7) var<storage, read_write> block_start: array<u32>;
@group(0) @binding(8) var<storage, read_write> sorted: array<u32>;
// Fixed-point transfer sums, ACC_FIELDS per cell, field-major.
@group(0) @binding(9) var<storage, read_write> acc: array<atomic<i32>>;
// Solver fields, CELL_FIELDS per cell, field-major.
@group(0) @binding(10) var<storage, read_write> cf: array<f32>;
// Per-block partial sums for the solver's dot products.
@group(0) @binding(11) var<storage, read_write> partial: array<vec4f>;
@group(0) @binding(12) var<storage, read_write> spray: array<Spray>;
// When water first touched each ground texel (f32 bits, only ever lowered) and where and
// when each spray slot landed (x, z, radius, t). The renderer rebuilds the wet ground for
// any point in time from these.
@group(0) @binding(13) var<storage, read_write> touch: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> landed: array<vec4f>;
// The last step's blocks, and each block of the domain's index among them when it had one.
// An index is only that block's if the list agrees.
@group(0) @binding(29) var<storage, read_write> prev_ptr: array<u32>;
@group(0) @binding(30) var<storage, read_write> block_list_prev: array<u32>;
// Indirect dispatch arguments, only bound to the kernels that write them: the step's own
// (see IND_*) or the pressure solver's (SOLVE_*), which are in a buffer of their own so the
// solver can switch itself off while being dispatched from the step's.
@group(1) @binding(0) var<storage, read_write> args: array<u32>;

const NONE: u32 = 0xffffffffu;
const FLT_MAX_BITS: u32 = 0x7f7fffffu;
const PENDING: u32 = 0xfffffffeu;
const RHO_AIR: f32 = 1.2;
const MU_AIR: f32 = 1.8e-5;
const SIGMA_W: f32 = 0.072;

// Transfer sums: momentum and weight of each of the three owned faces, the cell's own
// weight and weighted pressure, how many particles sit in it, the rubber's push on the
// water and the rubber area behind it, the bullets' change to each face's velocity, the
// volume their cavities open in the cell per second, and the area of the rubber whose
// vertices are in the cell, on water or not, and that times its velocity.
fn a_mom(d: u32) -> u32 {
    return 2u * d;
}
fn a_wt(d: u32) -> u32 {
    return 2u * d + 1u;
}
const A_WC: u32 = 6u;
const A_PC: u32 = 7u;
const A_N: u32 = 8u;
const A_MF: u32 = 9u;
const A_MA: u32 = 10u;
const A_BU: u32 = 11u;
const A_SRC: u32 = 14u;
const A_MW: u32 = 15u;
const A_MV: u32 = 16u;
const ACC_FIELDS: u32 = 19u;

// Fixed-point scales. A bulk cell sums about 8 particles, so these hold weights to about
// 100, momentum to a few km/s and pressure (in MPa) to a few GPa.
const FP_W: f32 = 16777216.0;
const FP_M: f32 = 65536.0;
const FP_P: f32 = 65536.0;
// The rubber's push in newtons, and its area in mm^2 (times m/s for A_MV).
const FP_F: f32 = 4194304.0;
const FP_A: f32 = 65536.0;
// Cavity volume in m^3/s.
const FP_Q: f32 = 16777216.0;

// Solver fields. The search direction has two, so an iteration can build the new one from
// the old while neighbours still read it.
const F_U: u32 = 0u;
const F_P: u32 = 3u;
const F_R: u32 = 4u;
const F_Z: u32 = 5u;
const F_D: u32 = 6u;
const F_Q: u32 = 8u;
const F_DIAG: u32 = 9u;
const F_TYPE: u32 = 10u;
const CELL_FIELDS: u32 = 11u;

// Cell types, stored as f32 in F_TYPE. Covered cells are gas behind rubber that lies on the
// water, which the water sees at the rubber's pressure. Rubber cells hold rubber with no
// water behind it. Both are walls to the air.
const GAS: f32 = 0.0;
const WATER: f32 = 1.0;
const SOLID: f32 = 2.0;
const COVERED: f32 = 3.0;
const RUBBER: f32 = 4.0;

fn cap_cells() -> u32 {
    return U.bdims.w * 64u;
}

fn h() -> f32 {
    return U.origin.w;
}

fn dt() -> f32 {
    return f32(st.dt_ns) * 1e-9;
}

fn now() -> f32 {
    return f32(st.t_ns) * 1e-9;
}

fn block_slot(b: vec3i) -> u32 {
    return (u32(b.x) * U.bdims.y + u32(b.y)) * U.bdims.z + u32(b.z);
}

fn slot_coord(s: u32) -> vec3i {
    return vec3i(vec3u(s / (U.bdims.z * U.bdims.y), (s / U.bdims.z) % U.bdims.y, s % U.bdims.z));
}

fn local_coord(i: u32) -> vec3i {
    let l = i % 64u;
    return vec3i(vec3u(l / 16u, (l / 4u) % 4u, l % 4u));
}

// Cell coordinates of the storage index `i`.
fn cell_coord(i: u32) -> vec3i {
    return slot_coord(block_list[i / 64u]) * 4 + local_coord(i);
}

// The cell `off` away from storage index `cell`, at most one block over, or NONE where no
// block is allocated. Only valid once this step's neighbour table is built.
fn neighbour(cell: u32, off: vec3i) -> u32 {
    let q = local_coord(cell) + off;
    let nb = select(vec3i(0), vec3i(1), q >= vec3i(4)) - select(vec3i(0), vec3i(1), q < vec3i(0));
    let b = block_nb[(cell / 64u) * 27u + u32((nb.x + 1) * 9 + (nb.y + 1) * 3 + nb.z + 1)];
    if (b == NONE) { return NONE; }
    let l = vec3u(q - nb * 4);
    return b * 64u + l.x * 16u + l.y * 4u + l.z;
}

fn acc_i(field: u32, cell: u32) -> u32 {
    return field * cap_cells() + cell;
}

fn cf_i(field: u32, cell: u32) -> u32 {
    return field * cap_cells() + cell;
}

fn cell_type(cell: u32) -> f32 {
    if (cell == NONE) { return GAS; }
    return cf[cf_i(F_TYPE, cell)];
}

fn to_fp(x: f32, scale: f32) -> i32 {
    return i32(round(clamp(x * scale, -2e9, 2e9)));
}

fn from_fp(x: i32, scale: f32) -> f32 {
    return f32(x) / scale;
}

// Where the lattice of component `d` sits inside a cell: faces are on the low side in their
// own dimension, cell centres (d = 3) in the middle.
fn lattice_offset(d: u32) -> vec3f {
    var o = vec3f(0.5);
    if (d < 3u) { o[d] = 0.0; }
    return o;
}

fn ground_touch(p: vec3f) {
    let hw = U.wet.x;
    let res = U.limits.x;
    let uv = (p.xz + hw) / (2.0 * hw);
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return; }
    let c = vec2u(uv * f32(res));
    atomicMin(&touch[c.y * res + c.x], bitcast<u32>(now()));
}
