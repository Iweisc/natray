// Rebuilds the state the renderer reads for a time between keyframes a and b. Positions
// blend linearly. Things that appear or vanish in between switch at the midpoint, except
// drops that land, which fly to their landing point and vanish there.

@group(0) @binding(0) var<uniform> C: CacheU;
@group(0) @binding(1) var<storage, read> key_a: array<u32>;
@group(0) @binding(2) var<storage, read> key_b: array<u32>;
@group(0) @binding(3) var<storage, read> spray_w: array<vec2f>;
@group(0) @binding(4) var<storage, read> landed: array<vec4f>;
@group(0) @binding(5) var<storage, read> touch: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_p: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> out_spray: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> out_mem: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> out_alive: array<u32>;
@group(0) @binding(10) var<storage, read_write> out_wet: array<atomic<u32>>;

fn get_a(base: u32) -> vec2u {
    return vec2u(key_a[base], key_a[base + 1u]);
}

fn get_b(base: u32) -> vec2u {
    return vec2u(key_b[base], key_b[base + 1u]);
}

fn kw(a: bool, w: u32) -> u32 {
    if (a) { return key_a[w]; }
    return key_b[w];
}

// Position and radius of spray slot i in key a or b, radius 0 where the slot is empty.
fn spray_slot(a: bool, i: u32) -> vec4f {
    let n = kw(a, 0u);
    if (i >= n) { return vec4f(0.0); }
    let nw = bit_words(n);
    let w = i / 32u;
    let bits = kw(a, spray_word() + w);
    let bit = 1u << (i % 32u);
    if ((bits & bit) == 0u) { return vec4f(0.0); }
    let base = spray_word() + 2u * nw + 3u * (kw(a, spray_word() + nw + w) + countOneBits(bits & (bit - 1u)));
    return vec4f(unpack_pos(vec2u(kw(a, base), kw(a, base + 1u))), bitcast<f32>(kw(a, base + 2u)));
}

@compute @workgroup_size(256)
fn show_water(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= C.counts.x) { return; }
    let wa = get_a(KEY_HEAD + 2u * i);
    let wb = get_b(KEY_HEAD + 2u * i);
    let pa = unpack_pos(wa);
    let pb = unpack_pos(wb);
    let f = C.time.y;
    var pos = select(pb, pa, f < 0.5);
    var live = select(flagged(wb), flagged(wa), f < 0.5);
    var vel = vec3f(0.0);
    if (flagged(wa) && flagged(wb)) {
        pos = mix(pa, pb, f);
        vel = (pb - pa) * C.time.z;
    }
    out_p[2u * i] = vec4f(pos, 1.0);
    out_p[2u * i + 1u] = vec4f(vel, select(0.0, 1.0, live));
}

@compute @workgroup_size(256)
fn show_mem(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= C.counts.y) { return; }
    let base = mem_word() + 2u * i;
    out_mem[i] = vec4f(mix(unpack_pos(get_a(base)), unpack_pos(get_b(base)), C.time.y), 0.0);
}

@compute @workgroup_size(256)
fn show_alive(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= C.counts.z) { return; }
    let w = alive_word() + t / 32u;
    let bits = select(key_b[w], key_a[w], C.time.y < 0.5);
    out_alive[t] = (bits >> (t % 32u)) & 1u;
}

@compute @workgroup_size(256)
fn show_spray(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    if (i >= max(key_a[0], key_b[0])) { return; }
    let sa = spray_slot(true, i);
    let sb = spray_slot(false, i);
    let pa = sa.xyz;
    let pb = sb.xyz;
    let ra = sa.w;
    let rb = sb.w;
    let f = C.time.y;
    let t = C.time.x;
    var pos = select(pb, pa, f < 0.5);
    var r = select(rb, ra, f < 0.5);
    var vel = vec3f(0.0);
    let l = landed[i];
    if (ra > 0.0 && rb > 0.0) {
        pos = mix(pa, pb, f);
        r = mix(ra, rb, f);
        vel = (pb - pa) * C.time.z;
    } else if (ra > 0.0 && l.z > 0.0 && l.w <= C.wet.z) {
        let ta = C.time.w;
        let hit = vec3f(l.x, l.z, l.y);
        let s = clamp((t - ta) / max(l.w - ta, 1e-9), 0.0, 1.0);
        pos = mix(pa, hit, s);
        r = select(0.0, ra, t < l.w);
        vel = (hit - pa) / max(l.w - ta, 1e-9);
    }
    // A negative weight hides a drop that is still part of the crown sheet.
    let w = spray_w[i];
    out_spray[2u * i] = vec4f(pos, r);
    out_spray[2u * i + 1u] = vec4f(vel, select(w.x, -w.x, t < w.y));
}

// A landing drop spreads to a few times its own diameter. Units are 1e-3 mm^3 per cell.
@compute @workgroup_size(256)
fn show_wet_drops(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    // Anything landed by now has a slot in key b.
    if (i >= key_b[0]) { return; }
    let l = landed[i];
    if (l.z <= 0.0 || l.w > C.time.x) { return; }
    let h = C.wet.x;
    let res = u32(C.wet.y);
    let cell = 2.0 * h / f32(res);
    let r = l.z;
    let k = i32(clamp(3.0 * r / cell, 0.0, 2.0));
    let n = f32((2 * k + 1) * (2 * k + 1));
    let amount = u32(4.18879 * r * r * r * 1e12 / n);
    for (var y = -k; y <= k; y++) {
        for (var x = -k; x <= k; x++) {
            let uv = (l.xy + vec2f(f32(x), f32(y)) * cell + h) / (2.0 * h);
            if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { continue; }
            let c = vec2u(uv * f32(res));
            atomicAdd(&out_wet[c.y * res + c.x], amount);
        }
    }
}

// Where the water body itself has touched down the ground is soaked.
@compute @workgroup_size(256)
fn show_wet_touch(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = flat_index(gid, nwg);
    let res = u32(C.wet.y);
    if (i >= res * res || C.time.x < 0.0) { return; }
    if (touch[i] <= bitcast<u32>(C.time.x)) { atomicMax(&out_wet[i], 4000000u); }
}
