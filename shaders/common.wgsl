const PI: f32 = 3.14159265358979;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    return pcg(a ^ pcg(b ^ pcg(c)));
}

fn rnd(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed >> 8u) * (1.0 / 16777216.0);
}

fn rnd_gauss(seed: ptr<function, u32>) -> f32 {
    let u1 = max(rnd(seed), 1e-7);
    let u2 = rnd(seed);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

fn ortho_basis(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    return mat3x3f(t, bt, n);
}

fn flat_index(gid: vec3u, nwg: vec3u) -> u32 {
    return gid.x + gid.y * nwg.x * 256u;
}
