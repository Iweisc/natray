// Keyframe layout shared by pack.wgsl and show.wgsl.
//
// A keyframe buffer holds, in words: the spray slot count N, the live spray count and two
// unused words, then water particles and membrane vertices (two words each), one alive bit
// per triangle, and the spray. Most slots are empty after their drop has landed, so the
// spray part only stores live ones: a live bit per slot, the number of live slots before
// each word of bits, and three words per live slot (position and radius) in slot order.
//
// Positions use 21 bits per axis inside an 8 m box, about 4 um steps. The top bit of the
// second word flags a live water particle.

struct CacheU {
    counts: vec4u,  // x particles, y membrane vertices, z triangles, w spray capacity
    time: vec4f,    // x shown time, y blend towards key b, z 1 / (tb - ta), w ta
    wet: vec4f,     // x half size, y resolution, z tb
}

const Q_LO: vec3f = vec3f(-4.0, -0.5, -4.0);
const Q_SIZE: f32 = 8.0;
const Q_MAX: f32 = 2097151.0;
const KEY_HEAD: u32 = 4u;

fn pack_pos(p: vec3f, flag: bool) -> vec2u {
    let q = vec3u(clamp((p - Q_LO) / Q_SIZE, vec3f(0.0), vec3f(1.0)) * Q_MAX + 0.5);
    return vec2u(q.x | (q.y << 21u), (q.y >> 11u) | (q.z << 10u) | select(0u, 1u << 31u, flag));
}

fn unpack_pos(w: vec2u) -> vec3f {
    let x = w.x & 0x1fffffu;
    let y = (w.x >> 21u) | ((w.y & 0x3ffu) << 11u);
    let z = (w.y >> 10u) & 0x1fffffu;
    return Q_LO + vec3f(f32(x), f32(y), f32(z)) * (Q_SIZE / Q_MAX);
}

fn flagged(w: vec2u) -> bool {
    return (w.y >> 31u) != 0u;
}

fn mem_word() -> u32 {
    return KEY_HEAD + 2u * C.counts.x;
}

fn alive_word() -> u32 {
    return KEY_HEAD + 2u * (C.counts.x + C.counts.y);
}

fn spray_word() -> u32 {
    return alive_word() + (C.counts.z + 31u) / 32u;
}

// Words of live bits for n spray slots.
fn bit_words(n: u32) -> u32 {
    return (n + 31u) / 32u;
}
