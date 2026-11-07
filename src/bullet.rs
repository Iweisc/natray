//! What the GPU needs of each bullet, see bullet.wgsl: its state where the flight through
//! the air hands it over, and its surface cut into patches.

use crate::flight::{profile, Body, Flight};
use bytemuck::{Pod, Zeroable};
use glam::{DQuat, DVec3};
use std::f64::consts::{PI, TAU};

pub const MAX_PATCHES: usize = 576;
pub const N_SEC: usize = 32;
pub const LOG_CAP: usize = 256;
const SLICES: usize = 32;
/// Points of the outline per bullet, see bullet.wgsl.
pub const OUTLINE: usize = SLICES + 1;
const SECTORS: usize = 16;

/// See Bullet in bullet.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuBullet {
    pub x: [f32; 4],
    pub v: [f32; 4],
    pub q: [f32; 4],
    pub l: [f32; 4],
    pub shape: [f32; 4],
    pub reach: [f32; 4],
    pub info: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPatch {
    p: [f32; 4],
    n: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PoseEntry {
    pub x: [f32; 4],
    pub q: [f32; 4],
    pub v: [f32; 4],
}

impl PoseEntry {
    pub fn axis(&self) -> DVec3 {
        let q = DQuat::from_xyzw(self.q[0] as f64, self.q[1] as f64, self.q[2] as f64, self.q[3] as f64);
        q * DVec3::Z
    }
}

fn v4(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

/// The bullet's outline from the base to the nose, as (u, r), at equal steps along it.
fn outline(b: &Body) -> Vec<(f64, f64)> {
    let fine = 4000;
    let fine_line: Vec<(f64, f64)> = (0..=fine).map(|i| b.length * i as f64 / fine as f64).map(|u| (u, profile(b.nose, b.radius, b.length, u))).collect();
    let mut s = vec![0.0];
    for w in fine_line.windows(2) {
        s.push(s.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
    }
    let total = *s.last().unwrap();
    (0..=SLICES)
        .map(|k| {
            let target = total * k as f64 / SLICES as f64;
            let i = s.partition_point(|&x| x < target).clamp(1, fine);
            let f = (target - s[i - 1]) / (s[i] - s[i - 1]).max(1e-15);
            let (a, c) = (fine_line[i - 1], fine_line[i]);
            (a.0 + (c.0 - a.0) * f, a.1 + (c.1 - a.1) * f)
        })
        .collect()
}

/// The outline in the body frame, (z, r) from the centre of mass, for contact.
pub fn gpu_outline(b: &Body) -> Vec<[f32; 2]> {
    outline(b).iter().map(|&(u, r)| [(u - b.u_cg) as f32, r as f32]).collect()
}

/// The bullet's surface in its own frame, centred on its centre of mass with the nose along
/// +z: bands of equal width along the outline, cut into sectors, and a disk wherever an end
/// is flat.
pub fn surface(b: &Body) -> Vec<GpuPatch> {
    let mut out = Vec::new();
    let mut ring = |u: f64, r: f64, radial: f64, axial: f64, area: f64| {
        for j in 0..SECTORS {
            let phi = (j as f64 + 0.5) * TAU / SECTORS as f64;
            let (c, sn) = (phi.cos(), phi.sin());
            out.push(GpuPatch { p: v4(DVec3::new(r * c, r * sn, u - b.u_cg), area / SECTORS as f64), n: v4(DVec3::new(radial * c, radial * sn, axial), 0.0) });
        }
    };
    for w in outline(b).windows(2) {
        let ((ua, ra), (ub, rb)) = (w[0], w[1]);
        let len = (ub - ua).hypot(rb - ra);
        if len <= 0.0 {
            continue;
        }
        let r = 0.5 * (ra + rb);
        ring(0.5 * (ua + ub), r, (ub - ua) / len, -(rb - ra) / len, TAU * r * len);
    }
    // A sector's centroid is about two thirds of the way out.
    for (u, axial) in [(0.0, -1.0), (b.length, 1.0)] {
        let r = profile(b.nose, b.radius, b.length, u);
        if r > 1e-9 {
            ring(u, 2.0 / 3.0 * r, 0.0, axial, PI * r * r);
        }
    }
    out
}

/// How far the surface gets from the centre of mass.
fn hull(b: &Body) -> f64 {
    outline(b).iter().map(|&(u, r)| (u - b.u_cg).hypot(r)).fold(0.0, f64::max)
}

/// The bullet as the flight through the air hands it over, joining the simulation at
/// `arrive_ns`.
pub fn initial(f: &Flight, arrive_ns: u32, patches: usize) -> GpuBullet {
    let b = &f.body;
    let (x, u, n, h, roll) = f.handoff();
    let q = DQuat::from_rotation_arc(DVec3::Z, n.normalize());
    GpuBullet {
        x: v4(x, b.mass),
        v: v4(u, roll),
        q: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
        l: v4(h, u.length()),
        shape: [b.ix as f32, b.iy as f32, b.length as f32, hull(b) as f32],
        reach: [0.0; 4],
        info: [arrive_ns, patches as u32, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Shot, BULLETS};

    #[test]
    fn surfaces_are_closed() {
        for bt in BULLETS {
            let body = Body::new(&Shot::with_bullet(bt));
            let ps = surface(&body);
            assert!(ps.len() <= MAX_PATCHES);
            let net: DVec3 = ps.iter().map(|p| DVec3::new(p.n[0] as f64, p.n[1] as f64, p.n[2] as f64) * p.p[3] as f64).sum();
            let area: f64 = ps.iter().map(|p| p.p[3] as f64).sum();
            println!("{}: {} patches, area {:.1} mm^2, net {:.2e}", bt.name, ps.len(), area * 1e6, net.length() / area);
            assert!(net.length() < 1e-3 * area);
            let line = gpu_outline(&body);
            assert_eq!(line.len(), OUTLINE);
            assert!(line.windows(2).all(|w| w[1][0] >= w[0][0]), "{}: contact needs z to rise along the outline", bt.name);
        }
    }
}
