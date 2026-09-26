// Compresses the live simulation into a keyframe.

struct SimParticle { pos: vec4f, vel: vec4f, c0: vec4f, c1: vec4f, c2: vec4f }
struct SimSpray { pos: vec4f, vel: vec4f, air: vec4f }

@group(0) @binding(0) var<uniform> C: CacheU;
@group(0) @binding(1) var<storage, read> sim_p: array<SimParticle>;
@group(0) @binding(2) var<storage, read> sim_mem: array<vec4f>;
@group(0) @binding(3) var<storage, read> sim_alive: array<u32>;
@group(0) @binding(4) var<storage, read> sim_spray: array<SimSpray>;
// Words 0 and 1 (spray slot and live counts) are written by the CPU before this runs.
@group(0) @binding(5) var<storage, read_write> key: array<u32>;
// Per spray slot: weight and when it tears loose from the crown sheet.
@group(0) @binding(6) var<storage, read_write> spray_w: array<vec2f>;
// The spray slot count is word 16.
@group(0) @binding(7) var<storage, read> sim_state: array<u32>;
// Live spray index, built before the keyframe is placed so its size is known: slot and live
// counts, two unused words, then the live bits and the live count before each bit word, both
// sized for the full capacity.
@group(0) @binding(8) var<storage, read_write> index: array<u32>;

fn prefix_word() -> u32 {
    return 4u + bit_words(C.counts.w);
}

var<workgroup> part: array<u32, 256>;

fn put(base: u32, w: vec2u) {
    key[base] = w.x;
    key[base + 1u] = w.y;
}

@compute @workgroup_size(256)
fn pack_water(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= C.counts.x) { return; }
    let p = sim_p[i];
    put(KEY_HEAD + 2u * i, pack_pos(p.pos.xyz, p.vel.w > 0.5));
}

@compute @workgroup_size(256)
fn pack_mem(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= C.counts.y) { return; }
    put(mem_word() + 2u * i, pack_pos(sim_mem[i].xyz, false));
}

@compute @workgroup_size(256)
fn pack_alive(@builtin(global_invocation_id) gid: vec3u) {
    let w = gid.x;
    if (w * 32u >= C.counts.z) { return; }
    var bits = 0u;
    for (var k = 0u; k < 32u; k++) {
        let t = w * 32u + k;
        if (t < C.counts.z && sim_alive[t] != 0u) { bits |= 1u << k; }
    }
    key[alive_word() + w] = bits;
}

@compute @workgroup_size(256)
fn index_bits(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let w = flat_index(gid, nwg);
    let n = min(sim_state[16], C.counts.w);
    if (w == 0u) { index[0] = n; }
    if (w >= bit_words(n)) { return; }
    var bits = 0u;
    for (var k = 0u; k < 32u; k++) {
        let i = w * 32u + k;
        if (i < n && sim_spray[i].pos.w > 0.0) { bits |= 1u << k; }
    }
    index[4u + w] = bits;
}

// Exclusive prefix sum of the live counts, in one workgroup: each thread sums a run of
// words, the run totals are scanned, then each thread writes its run.
@compute @workgroup_size(256)
fn index_scan(@builtin(local_invocation_index) li: u32) {
    let nw = bit_words(min(sim_state[16], C.counts.w));
    let chunk = (nw + 255u) / 256u;
    let lo = min(li * chunk, nw);
    let hi = min(lo + chunk, nw);
    var sum = 0u;
    for (var w = lo; w < hi; w++) {
        sum += countOneBits(index[4u + w]);
    }
    part[li] = sum;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off *= 2u) {
        var v = part[li];
        if (li >= off) { v += part[li - off]; }
        workgroupBarrier();
        part[li] = v;
        workgroupBarrier();
    }
    var run = part[li] - sum;
    for (var w = lo; w < hi; w++) {
        index[prefix_word() + w] = run;
        run += countOneBits(index[4u + w]);
    }
    if (li == 255u) { index[1] = part[255]; }
}

@compute @workgroup_size(256)
fn pack_spray(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    let n = key[0];
    if (i >= n) { return; }
    let nw = bit_words(n);
    let w = i / 32u;
    let bits = index[4u + w];
    let before = index[prefix_word() + w];
    if (i % 32u == 0u) {
        key[spray_word() + w] = bits;
        key[spray_word() + nw + w] = before;
    }
    let s = sim_spray[i];
    // A slot keeps both for life, so the latest values are right for every keyframe.
    spray_w[i] = vec2f(s.vel.w, s.air.w);
    let bit = 1u << (i % 32u);
    if ((bits & bit) == 0u) { return; }
    let base = spray_word() + 2u * nw + 3u * (before + countOneBits(bits & (bit - 1u)));
    put(base, pack_pos(s.pos.xyz, false));
    key[base + 2u] = bitcast<u32>(s.pos.w);
}
