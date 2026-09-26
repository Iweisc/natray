// Sparse water density lookup, shared by the ray tracer and the sprites.
// Needs W, vindex, dens_tex and lin_samp declared by the including shader.

fn water_slot(b: vec3i) -> u32 {
    if (any(b < vec3i(0)) || any(b >= vec3i(W.dims.xyz))) { return 0u; }
    return vindex[(u32(b.x) * W.dims.y + u32(b.y)) * W.dims.z + u32(b.z)];
}

fn dens(p: vec3f) -> f32 {
    let c = (p - W.origin.xyz) / W.cell.x;
    let b = floor(c / 8.0);
    let s = water_slot(vec3i(b));
    if (s == 0u || s == BRICK_MARKED) { return 0.0; }
    let slot = s - 1u;
    let so = vec3f(f32(slot % ATLAS_B), f32((slot / ATLAS_B) % ATLAS_B), f32(slot / (ATLAS_B * ATLAS_B))) * 9.0;
    return textureSampleLevel(dens_tex, lin_samp, (so + (c - b * 8.0) + 0.5) / ATLAS, 0.0).r;
}
