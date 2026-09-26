// Per-frame acceleration structures: water density grid, membrane triangle grid,
// mist extinction grid and the caustic map placement.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> P: array<Particle>;
@group(0) @binding(2) var<storage, read_write> rs: array<atomic<i32>, 32>;
@group(0) @binding(3) var<storage, read_write> wg: WGrid;
@group(0) @binding(4) var<storage, read_write> dens_acc: array<atomic<u32>>;
@group(0) @binding(5) var dens_img: texture_storage_3d<r32float, write>;
// Per atlas slot: highest and lowest density in the brick (f32 bits, densities are >= 0).
@group(0) @binding(6) var<storage, read_write> brick_range: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> vindex: array<atomic<u32>>;
@group(1) @binding(0) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(8) var<storage, read> mpos: array<vec4f>;
@group(0) @binding(9) var<storage, read> tris: array<Tri>;
@group(0) @binding(10) var<storage, read> alive: array<u32>;
@group(0) @binding(11) var<storage, read> vtri_off: array<u32>;
@group(0) @binding(12) var<storage, read> vtri: array<u32>;
@group(0) @binding(13) var<storage, read_write> mnorm: array<vec4f>;
@group(0) @binding(14) var<storage, read_write> mg: UGrid;
@group(0) @binding(15) var<storage, read_write> mem_index: array<u32>;
@group(0) @binding(16) var<storage, read> spray: array<Spray>;
@group(0) @binding(17) var<storage, read_write> mist_acc: array<atomic<u32>>;
@group(0) @binding(18) var mist_img: texture_storage_3d<rgba16float, write>;
@group(0) @binding(19) var<storage, read_write> sg: UGrid;
@group(0) @binding(20) var<storage, read_write> cm: CMap;
@group(0) @binding(21) var<storage, read_write> sm_acc: array<atomic<i32>>;
@group(0) @binding(22) var<storage, read_write> spos: array<vec4f>;
@group(0) @binding(23) var<storage, read_write> aniso: array<vec4f>;
@group(0) @binding(24) var<storage, read_write> slot_brick: array<u32>;
@group(0) @binding(25) var<storage, read_write> mist_blur: array<f32>;

const MIST_MAX: u32 = 128u;
const POOL: u32 = ATLAS_B * ATLAS_B * ATLAS_B;
const DENS_SCALE: f32 = 1000.0;
const MIST_SCALE: f32 = 100.0;
const SM_MAX: u32 = 128u;
const SM_FP: f32 = 65536.0;

fn grow(base: u32, p: vec3f) {
    let q = vec3i(round(p * FIXP));
    atomicMin(&rs[base], q.x);
    atomicMin(&rs[base + 1u], q.y);
    atomicMin(&rs[base + 2u], q.z);
    atomicMax(&rs[base + 4u], q.x);
    atomicMax(&rs[base + 5u], q.y);
    atomicMax(&rs[base + 6u], q.z);
}

fn box_lo(base: u32) -> vec3f {
    return vec3f(f32(atomicLoad(&rs[base])), f32(atomicLoad(&rs[base + 1u])), f32(atomicLoad(&rs[base + 2u]))) / FIXP;
}
fn box_hi(base: u32) -> vec3f {
    return vec3f(f32(atomicLoad(&rs[base + 4u])), f32(atomicLoad(&rs[base + 5u])), f32(atomicLoad(&rs[base + 6u]))) / FIXP;
}

@compute @workgroup_size(256)
fn bounds_water(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.x || P[i].vel.w < 0.5) { return; }
    grow(0u, P[i].pos.xyz);
}

@compute @workgroup_size(256)
fn mem_normals(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= F.counts.z) { return; }
    var n = vec3f(0.0);
    var any_alive = false;
    for (var k = vtri_off[i]; k < vtri_off[i + 1u]; k++) {
        let t = vtri[k];
        if (alive[t] == 0u) { continue; }
        any_alive = true;
        let tr = tris[t];
        let p0 = mpos[tr.v0].xyz;
        n += cross(mpos[tr.v1].xyz - p0, mpos[tr.v2].xyz - p0);
    }
    let l = length(n);
    mnorm[i] = vec4f(select(vec3f(0.0, 1.0, 0.0), n / l, l > 1e-14), select(0.0, 1.0, any_alive));
    if (any_alive) { grow(8u, mpos[i].xyz); }
}

@compute @workgroup_size(256)
fn bounds_mist(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.y) { return; }
    let s = spray[i];
    if (s.pos.w <= 0.0 || s.vel.w <= 1.0) { return; }
    grow(16u, s.pos.xyz);
}

@compute @workgroup_size(1)
fn setup() {
    let wlo = box_lo(0u);
    let whi = box_hi(0u);
    let water_ok = whi.x >= wlo.x;
    wg.dims.w = select(0u, 1u, water_ok);
    if (water_ok) {
        let ext = whi - wlo;
        let s_cell = max(2.0 * F.water.y, max(ext.x, max(ext.y, ext.z)) / f32(SM_MAX - 6u));
        wg.s_origin = vec4f(wlo - 2.0 * s_cell, s_cell);
        wg.s_dims = vec4u(min(vec3u(ceil(ext / s_cell)) + 5u, vec3u(SM_MAX)), 1u);
        wg.lo = vec4f(wlo, 0.0);
        wg.hi = vec4f(whi, 0.0);
    }

    let mlo = box_lo(8u);
    let mhi = box_hi(8u);
    let mem_ok = mhi.x >= mlo.x;
    mg.origin = vec4f(mlo, 0.0);
    mg.dims = vec4u(0u, 0u, 0u, select(0u, 1u, mem_ok));

    let slo = box_lo(16u);
    let shi = box_hi(16u);
    if (shi.x >= slo.x) {
        let lo = slo - 0.015;
        let ext = shi + 0.015 - lo;
        let cs = max(max(ext.x, max(ext.y, ext.z)) / f32(MIST_MAX - 2u), 0.0015);
        let dims = min(vec3u(ceil(ext / cs)) + 1u, vec3u(MIST_MAX));
        sg.origin = vec4f(lo, cs);
        sg.dims = vec4u(dims, 1u);
        indirect[3] = (dims.x + 3u) / 4u;
        indirect[4] = (dims.y + 3u) / 4u;
        indirect[5] = (dims.z + 3u) / 4u;
    } else {
        sg.dims = vec4u(0u);
        indirect[3] = 0u;
        indirect[4] = 0u;
        indirect[5] = 0u;
    }

    // Caustic map: a sun-facing square emitter around everything that bends or blocks light,
    // projected onto the ground as a rectangle.
    let L = F.sun_dir.xyz;
    var lo = select(vec3f(1e9), wlo, water_ok);
    var hi = select(vec3f(-1e9), whi, water_ok);
    if (mem_ok) {
        lo = min(lo, mlo);
        hi = max(hi, mhi);
    }
    if ((water_ok || mem_ok) && L.y > 0.05) {
        let c = 0.5 * (lo + hi);
        let r = 0.5 * length(hi - lo) * 1.35 + 0.02;
        var u = vec3f(L.x, 0.0, L.z);
        u = select(vec3f(1.0, 0.0, 0.0), normalize(u), length(u) > 1e-4);
        let v = cross(vec3f(0.0, 1.0, 0.0), u);
        let g = c - L * (c.y / L.y);
        cm.center = vec4f(c, r);
        cm.u_axis = vec4f(u, r / L.y);
        cm.v_axis = vec4f(v, r);
        cm.ground = vec4f(g.x, 0.0, g.z, 1.0);
    } else {
        cm.ground = vec4f(0.0);
    }
}

fn bspline(fx: vec3f) -> array<vec3f, 3> {
    return array<vec3f, 3>(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}

// Per node: weight, first moment (3) and second moment (6), all relative to the node.
const SM_CH: u32 = 10u;

// The stats grid is coarse already, so a trilinear stencil resolves it well enough and
// costs a third of the atomics a quadratic one does.
fn hat(fx: vec3f) -> array<vec3f, 2> {
    return array<vec3f, 2>(vec3f(1.0) - fx, fx);
}

fn sm_index(n: vec3i) -> u32 {
    let c = vec3u(clamp(n, vec3i(0), vec3i(wg.s_dims.xyz) - 1));
    return ((c.x * SM_MAX + c.y) * SM_MAX + c.z) * SM_CH;
}

fn sm_load(idx: u32) -> f32 {
    return f32(atomicLoad(&sm_acc[idx])) / SM_FP;
}

fn outer(a: vec3f, b: vec3f) -> mat3x3f {
    return mat3x3f(a * b.x, a * b.y, a * b.z);
}

fn inv3(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]);
    let r1 = cross(m[2], m[0]);
    let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}

// Neighbourhood statistics after Yu & Turk, gathered through a coarse grid: every particle
// scatters its offset moments to the nodes, then reads back the weighted mean and
// covariance of the particles around it.
@compute @workgroup_size(256)
fn splat_smooth(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.x || wg.dims.w == 0u) { return; }
    let p = P[i];
    if (p.vel.w < 0.5) { return; }
    let xp = (p.pos.xyz - wg.s_origin.xyz) / wg.s_origin.w;
    let base = vec3i(floor(xp));
    let fx = xp - vec3f(base);
    let w = hat(fx);
    for (var a = 0; a < 2; a++) {
        for (var b = 0; b < 2; b++) {
            for (var c = 0; c < 2; c++) {
                let off = vec3i(a, b, c);
                let weight = w[a].x * w[b].y * w[c].z * SM_FP;
                let d = fx - vec3f(off);
                let dw = d * weight;
                let idx = sm_index(base + off);
                atomicAdd(&sm_acc[idx], i32(round(weight)));
                atomicAdd(&sm_acc[idx + 1u], i32(round(dw.x)));
                atomicAdd(&sm_acc[idx + 2u], i32(round(dw.y)));
                atomicAdd(&sm_acc[idx + 3u], i32(round(dw.z)));
                atomicAdd(&sm_acc[idx + 4u], i32(round(dw.x * d.x)));
                atomicAdd(&sm_acc[idx + 5u], i32(round(dw.y * d.y)));
                atomicAdd(&sm_acc[idx + 6u], i32(round(dw.z * d.z)));
                atomicAdd(&sm_acc[idx + 7u], i32(round(dw.x * d.y)));
                atomicAdd(&sm_acc[idx + 8u], i32(round(dw.x * d.z)));
                atomicAdd(&sm_acc[idx + 9u], i32(round(dw.y * d.z)));
            }
        }
    }
}

// Writes the smoothed position and the square root of the kernel shape matrix. Bulk water
// gets a round kernel, surfaces a flattened one, sheets a thin disc and threads a needle,
// so thin sheets render as continuous films instead of rows of beads.
@compute @workgroup_size(256)
fn smooth_positions(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.x || wg.dims.w == 0u) { return; }
    let p = P[i];
    if (p.vel.w < 0.5) { return; }
    let cell = wg.s_origin.w;
    let xp = (p.pos.xyz - wg.s_origin.xyz) / cell;
    let base = vec3i(floor(xp));
    let fx = xp - vec3f(base);
    let w = hat(fx);
    var den = 0.0;
    var m1 = vec3f(0.0);
    var m2 = mat3x3f();
    for (var a = 0; a < 2; a++) {
        for (var b = 0; b < 2; b++) {
            for (var c = 0; c < 2; c++) {
                let off = vec3i(a, b, c);
                let weight = w[a].x * w[b].y * w[c].z;
                let idx = sm_index(base + off);
                let ws = sm_load(idx);
                let ds = vec3f(sm_load(idx + 1u), sm_load(idx + 2u), sm_load(idx + 3u));
                let xy = sm_load(idx + 7u);
                let xz = sm_load(idx + 8u);
                let yz = sm_load(idx + 9u);
                let ms = mat3x3f(sm_load(idx + 4u), xy, xz, xy, sm_load(idx + 5u), yz, xz, yz, sm_load(idx + 6u));
                // Moments were stored relative to the node. Shift them to this particle.
                let r = vec3f(off) - fx;
                let dr = outer(ds, r);
                den += weight * ws;
                m1 += weight * (ds + ws * r);
                m2 += (ms + dr + transpose(dr) + outer(r, r) * ws) * weight;
            }
        }
    }
    let mu = m1 / max(den, 1e-6);
    // Two convolved linear hats have a variance of 1/3 cell^2 per axis, which is what a
    // full, uniform neighbourhood measures. Normalise so that bulk water gives I.
    let a = (m2 * (1.0 / max(den, 1e-6)) - outer(mu, mu)) * 3.0;
    let tr = a[0].x + a[1].y + a[2].z;
    var y = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    // Lone particles have no shape to follow and are becoming spray anyway.
    if (tr > 0.3) {
        // The regularisation caps the stretch at about 4:1 and keeps the thin axis at least
        // 1.5 density cells wide, so the grid can still resolve it.
        let h = F.water.z;
        let thin = 1.5 * min(wg.cell.x, min(wg.cell.y, wg.cell.z)) / h;
        let eps = max(0.07 * tr, thin * thin);
        let areg = (a + y * eps) * (1.0 / (1.0 + 0.07 * 3.0));
        // Denman-Beavers iteration for the matrix square root.
        var z = y;
        y = areg;
        for (var k = 0; k < 8; k++) {
            let yi = inv3(y);
            y = (y + inv3(z)) * 0.5;
            z = (z + yi) * 0.5;
        }
    }
    let ratio = 2.0 * F.water.y / cell;
    let lambda = 0.9 * ratio * ratio;
    spos[i] = vec4f(p.pos.xyz + lambda * cell * mu, 1.0);
    aniso[2u * i] = vec4f(y[0].x, y[1].y, y[2].z, y[0].y);
    aniso[2u * i + 1u] = vec4f(y[0].z, y[1].z, 0.0, 0.0);
}

fn kernel_shape(i: u32) -> mat3x3f {
    let a0 = aniso[2u * i];
    let a1 = aniso[2u * i + 1u];
    return mat3x3f(a0.x, a0.w, a1.x, a0.w, a0.y, a1.y, a1.x, a1.y, a0.z);
}

fn brick_index(b: vec3i) -> u32 {
    return (u32(b.x) * wg.dims.y + u32(b.y)) * wg.dims.z + u32(b.z);
}

// Marks every brick the particle's kernel reaches, widened by the resolve blur and by one
// more cell below so the brick whose extra texel row sees this water exists too.
@compute @workgroup_size(256)
fn mark_bricks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.x || P[i].vel.w < 0.5) { return; }
    let y = kernel_shape(i);
    let ext = wg.cell.w * vec3f(length(y[0]), length(y[1]), length(y[2]));
    let g = (spos[i].xyz - wg.origin.xyz) / wg.cell.x;
    let ec = ext / wg.cell.x;
    let bd = vec3i(wg.dims.xyz) - 1;
    let lo = clamp(vec3i(floor((ceil(g - ec) - 2.0) / 8.0)), vec3i(0), bd);
    let hi = clamp(vec3i(floor((floor(g + ec) + 1.0) / 8.0)), vec3i(0), bd);
    for (var x = lo.x; x <= hi.x; x++) {
        for (var yy = lo.y; yy <= hi.y; yy++) {
            for (var z = lo.z; z <= hi.z; z++) {
                atomicStore(&vindex[brick_index(vec3i(x, yy, z))], BRICK_MARKED);
            }
        }
    }
}

@compute @workgroup_size(256)
fn alloc_bricks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= wg.dims.x * wg.dims.y * wg.dims.z) { return; }
    if (atomicLoad(&vindex[i]) != BRICK_MARKED) { return; }
    let slot = u32(atomicAdd(&rs[25], 1));
    // A full pool drops bricks. At the default resolution a splash spread flat over a
    // square metre needs about 9000 of the 32768.
    if (slot >= POOL) {
        atomicStore(&vindex[i], 0u);
        return;
    }
    atomicStore(&vindex[i], slot + 1u);
    slot_brick[slot] = i;
    atomicStore(&brick_range[2u * slot], 0u);
    atomicStore(&brick_range[2u * slot + 1u], 0x7f800000u);
    for (var k = 0u; k < 512u; k++) {
        atomicStore(&dens_acc[slot * 512u + k], 0u);
    }
}

fn acc_index(c: vec3i) -> u32 {
    let b = c >> vec3u(3u);
    if (any(b < vec3i(0)) || any(b >= vec3i(wg.dims.xyz))) { return NO_CELL; }
    let s = atomicLoad(&vindex[brick_index(b)]);
    if (s == 0u) { return NO_CELL; }
    let l = vec3u(c & vec3i(7));
    return (s - 1u) * 512u + (l.x * 8u + l.y) * 8u + l.z;
}

const NO_CELL: u32 = 0xffffffffu;

@compute @workgroup_size(256)
fn splat_water(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.x || P[i].vel.w < 0.5) { return; }
    let xs = spos[i].xyz;
    let y = kernel_shape(i);
    // Kernel |G d| < h with G = y^-1. Scaling by det(G) keeps the splatted volume.
    let g = inv3(y);
    let norm = 1.0 / determinant(y);
    let rk = wg.cell.w;
    let cell = wg.cell.x;
    let ext = rk * vec3f(length(y[0]), length(y[1]), length(y[2]));
    let gp = (xs - wg.origin.xyz) / cell;
    let ec = ext / cell;
    let lo = vec3i(ceil(gp - ec));
    let hi = vec3i(floor(gp + ec));
    let inv_r2 = 1.0 / (rk * rk);
    for (var x = lo.x; x <= hi.x; x++) {
        for (var yy = lo.y; yy <= hi.y; yy++) {
            for (var z = lo.z; z <= hi.z; z++) {
                let c = vec3i(x, yy, z);
                let d = g * (wg.origin.xyz + vec3f(c) * cell - xs);
                let q = 1.0 - dot(d, d) * inv_r2;
                if (q > 0.0) {
                    let idx = acc_index(c);
                    if (idx != NO_CELL) { atomicAdd(&dens_acc[idx], u32(q * q * q * norm * DENS_SCALE)); }
                }
            }
        }
    }
}

fn acc_at(c: vec3i) -> f32 {
    let idx = acc_index(c);
    if (idx == NO_CELL) { return 0.0; }
    return f32(atomicLoad(&dens_acc[idx]));
}

// One workgroup per allocated brick: blur, normalise and store 9^3 texels, and remember the
// brick maximum for empty-space skipping.
@compute @workgroup_size(64)
fn resolve_water(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let slot = wid.x;
    if (slot >= min(u32(atomicLoad(&rs[25])), POOL)) { return; }
    let vb = slot_brick[slot];
    let bd = wg.dims.xyz;
    let b = vec3i(vec3u(vb / (bd.y * bd.z), (vb / bd.z) % bd.y, vb % bd.z));
    let so = vec3u(slot % ATLAS_B, (slot / ATLAS_B) % ATLAS_B, slot / (ATLAS_B * ATLAS_B)) * 9u;
    var m = 0.0;
    var lo = 1e30;
    for (var k = li; k < 729u; k += 64u) {
        let l = vec3u(k / 81u, (k / 9u) % 9u, k % 9u);
        let c = b * 8 + vec3i(l);
        var d = 0.4 * acc_at(c) + 0.1 * (acc_at(c + vec3i(1, 0, 0)) + acc_at(c - vec3i(1, 0, 0))
            + acc_at(c + vec3i(0, 1, 0)) + acc_at(c - vec3i(0, 1, 0))
            + acc_at(c + vec3i(0, 0, 1)) + acc_at(c - vec3i(0, 0, 1)));
        d /= DENS_SCALE * F.time.w;
        textureStore(dens_img, so + l, vec4f(d, 0.0, 0.0, 0.0));
        m = max(m, d);
        lo = min(lo, d);
    }
    atomicMax(&brick_range[2u * slot], bitcast<u32>(m));
    atomicMin(&brick_range[2u * slot + 1u], bitcast<u32>(lo));
}

// Index buffer for the membrane's ray tracing acceleration structure. Torn triangles
// collapse to a point, which rays never hit.
@compute @workgroup_size(256)
fn mem_indices(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= F.flags.w) { return; }
    let tr = tris[t];
    let live = alive[t] != 0u;
    mem_index[3u * t] = tr.v0;
    mem_index[3u * t + 1u] = select(tr.v0, tr.v1, live);
    mem_index[3u * t + 2u] = select(tr.v0, tr.v2, live);
}

@compute @workgroup_size(256)
fn splat_mist(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= F.counts.y || sg.dims.w == 0u) { return; }
    let s = spray[i];
    if (s.pos.w <= 0.0 || s.vel.w <= 1.0) { return; }
    let cs = sg.origin.w;
    // Mie extinction efficiency is ~2 for droplets much larger than the wavelength.
    let sigma = s.vel.w * 2.0 * PI * s.pos.w * s.pos.w / (cs * cs * cs);
    let g = (s.pos.xyz - sg.origin.xyz) / cs;
    let i0 = vec3i(floor(g));
    let f = g - vec3f(i0);
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let c = i0 + o;
        if (any(c < vec3i(0)) || any(c >= vec3i(sg.dims.xyz))) { continue; }
        let wv = select(1.0 - f, f, vec3<bool>(o.x == 1, o.y == 1, o.z == 1));
        let idx = (u32(c.x) * MIST_MAX + u32(c.y)) * MIST_MAX + u32(c.z);
        atomicAdd(&mist_acc[idx], u32(sigma * wv.x * wv.y * wv.z * MIST_SCALE));
    }
}

fn mist_at(c: vec3i) -> f32 {
    if (any(c < vec3i(0)) || any(c >= vec3i(sg.dims.xyz))) { return 0.0; }
    return f32(atomicLoad(&mist_acc[(u32(c.x) * MIST_MAX + u32(c.y)) * MIST_MAX + u32(c.z)]));
}

@compute @workgroup_size(4, 4, 4)
fn resolve_mist(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= sg.dims.xyz)) { return; }
    let c = vec3i(gid);
    var d = 0.4 * mist_at(c) + 0.1 * (mist_at(c + vec3i(1, 0, 0)) + mist_at(c - vec3i(1, 0, 0))
        + mist_at(c + vec3i(0, 1, 0)) + mist_at(c - vec3i(0, 1, 0))
        + mist_at(c + vec3i(0, 0, 1)) + mist_at(c - vec3i(0, 0, 1)));
    mist_blur[(gid.x * MIST_MAX + gid.y) * MIST_MAX + gid.z] = d / MIST_SCALE;
}

fn blur_at(c: vec3i) -> f32 {
    if (any(c < vec3i(0)) || any(c >= vec3i(sg.dims.xyz))) { return 0.0; }
    return mist_blur[(u32(c.x) * MIST_MAX + u32(c.y)) * MIST_MAX + u32(c.z)];
}

fn blur_lerp(g: vec3f) -> f32 {
    let i0 = vec3i(floor(g));
    let f = g - vec3f(i0);
    var v = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let w = select(1.0 - f, f, vec3<bool>(o.x == 1, o.y == 1, o.z == 1));
        v += blur_at(i0 + o) * w.x * w.y * w.z;
    }
    return v;
}

// Self-shadowing of the mist towards the sun, per voxel here instead of per march sample
// in the trace. Multiple scattering after Wrenninge et al.: octave k is 0.5^k as bright,
// sees 0.5^k of the optical depth and blends 0.5^k of the way from an isotropic phase to
// the real one. The trace only knows the phase, so this stores the two sums it needs:
// y weights the isotropic phase and z the difference to the real one. Dense mist would
// come out grey without the octaves.
@compute @workgroup_size(4, 4, 4)
fn mist_shadow(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= sg.dims.xyz)) { return; }
    let g = vec3f(gid);
    var od = 0.0;
    for (var k = 1; k <= 3; k++) {
        od += blur_lerp(g + F.sun_dir.xyz * f32(k) * 0.012 / sg.origin.w) * 0.012;
    }
    var iso = 0.0;
    var aniso = 0.0;
    var a = 1.0;
    for (var k = 0; k < 4; k++) {
        let e = a * exp(-od * a);
        iso += e;
        aniso += e * a;
        a *= 0.5;
    }
    let sigma = blur_at(vec3i(gid));
    // Light the droplets scatter among themselves on top of the sky light.
    let amb = 1.0 + 0.9 * (1.0 - exp(-sigma * 0.02));
    textureStore(mist_img, gid, vec4f(sigma, iso, aniso, amb));
}
