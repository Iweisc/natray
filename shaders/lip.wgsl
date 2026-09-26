// The cavity lip at the entry and exit holes, shared by the spray emitter and the crown
// sheet. src/lip.rs mirrors cavity(), lip_velocity() and mist_share() for the emission
// schedule.

const LIP_FINGERS: u32 = 22u;

// Radius and wall speed of the cavity `tau` after the bullet passed a slab with source
// strength `k`. a da/dt = K starts at k and the underpressure `p` takes p / (1.5 rho) per
// second off it while air vents in over `tv`. Past the widest point K is negative and the
// cavity closes.
fn cavity(k: f32, tau: f32, rb: f32, p: f32, tv: f32) -> vec2f {
    let b = p * tv / 1500.0;
    let e = exp(-tau / tv);
    let a = sqrt(max(rb * rb + 2.0 * (k * tau - b * (tau - tv * (1.0 - e))), 0.0));
    return vec2f(a, (k - b * (1.0 - e)) / max(a, 1e-6));
}

// Velocity (radial, along the axis) of the sheet the lip throws off while its wall moves
// out at `wall`. Radially a bit faster than the wall, so the sheet opens into a crown about
// 28 degrees off the axis. An estimate, not a measurement.
fn lip_velocity(wall: f32) -> vec2f {
    return wall * vec2f(1.15, 2.2);
}

// How long a slice of the sheet lasts before it tears into drops. A fast sheet flaps and
// tears within a centimetre, a slow one stretches a few before its rim eats it.
fn sheet_life(speed: f32) -> f32 {
    let s = max(speed, 0.5);
    return min(clamp(0.04 * sqrt(5.0 / s), 0.008, 0.04) / s, 0.006);
}

// Share of the lip material that the air shreds into mist rather than drops.
fn mist_share(speed: f32) -> f32 {
    return 0.04 + 0.8 * smoothstep(30.0, 90.0, speed);
}

// How far the balloon surface falls away from the tangent plane at the hole, `a` off axis.
fn lip_sag(a: f32, radius: f32) -> f32 {
    return radius - sqrt(max(radius * radius - a * a, 0.0));
}

// The rim breaks into fingers that stay put for the whole shot: centre angle, strength and
// width of finger j at one hole, irregularly spaced.
fn finger_at(mouth: u32, j: u32) -> vec3f {
    var s = hash3(j, mouth, 0xf1u);
    let w = 2.0 * PI / f32(LIP_FINGERS);
    return vec3f((f32(j) + 0.5 + 0.6 * (rnd(&s) - 0.5)) * w, mix(0.5, 1.0, rnd(&s)), w * mix(0.18, 0.3, rnd(&s)));
}

// 0 between fingers, up to 1 on one. Where it is high the sheet lasts longer, so its edge
// reaches out into the finger.
fn finger(phi: f32, mouth: u32) -> f32 {
    let n = i32(LIP_FINGERS);
    let w = 2.0 * PI / f32(LIP_FINGERS);
    let j0 = i32(floor(phi / w));
    var f = 0.0;
    for (var d = -1; d <= 1; d++) {
        let m = j0 + d;
        let j = (m % n + n) % n;
        let c = finger_at(mouth, u32(j));
        let x = (phi - c.x - f32(m - j) * w) / c.z;
        f = max(f, c.y * exp(-x * x));
    }
    return f;
}
