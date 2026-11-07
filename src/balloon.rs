//! CPU-side scene construction: the balloon's latex, the shape it takes filled with water and
//! hanging from its knot, its mesh and the water particles. The bullets are in flight.rs.

use crate::params::SimParams;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};
use std::collections::HashMap;
use std::f64::consts::PI;

pub const RHO_WATER: f64 = 1000.0;
pub const RHO_RUBBER: f64 = 950.0;

/// Natural rubber: Ogden's three-term fit to Treloar's measurements, as (mu in Pa, alpha).
/// membrane.wgsl has the same exponents.
pub const OGDEN: [(f64, f64); 3] = [(0.63e6, 1.3), (0.0012e6, 5.0), (-0.01e6, -2.0)];

/// The uninflated balloon is a sphere, open at the top where the neck is. The neck is this
/// wide relative to the sphere.
const NECK: f64 = 0.25;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuTri {
    pub v: [u32; 3],
    pub rest_area: f32,
    /// Inverse of the matrix of the two rest edges from v[0], in the triangle's own plane,
    /// row-major.
    pub dm_inv: [f32; 4],
}

/// A latex sheet: Ogden rubber, incompressible, in plane stress.
#[derive(Clone, Copy)]
pub struct Latex {
    pub mu: [f64; 3],
    pub thickness: f64,
}

impl Latex {
    /// OGDEN scaled to the given small-strain shear modulus.
    pub fn new(shear_modulus: f64, thickness: f64) -> Self {
        let g0 = OGDEN.iter().map(|&(m, a)| m * a).sum::<f64>() / 2.0;
        Self { mu: OGDEN.map(|(m, _)| m * shear_modulus / g0), thickness }
    }

    fn sigma(&self, a: f64, b: f64) -> f64 {
        self.mu.iter().zip(OGDEN).map(|(m, (_, al))| m * (a.powf(al) - b.powf(al))).sum()
    }

    /// Principal Cauchy stresses at stretches l1 and l2. A sheet can't be squeezed in its
    /// plane: it wrinkles instead, which leaves it pulled one way only, or slack.
    pub fn stress(&self, l1: f64, l2: f64) -> (f64, f64) {
        let (major, minor) = (l1.max(l2), l1.min(l2));
        if major <= 1.0 {
            return (0.0, 0.0);
        }
        let natural = major.powf(-0.5);
        if minor < natural {
            let s = self.sigma(major, natural);
            return if l1 >= l2 { (s, 0.0) } else { (0.0, s) };
        }
        let l3 = 1.0 / (l1 * l2);
        (self.sigma(l1, l3), self.sigma(l2, l3))
    }

    /// Force per length of the stretched sheet.
    pub fn tension(&self, l1: f64, l2: f64) -> (f64, f64) {
        let (s1, s2) = self.stress(l1, l2);
        let t = self.thickness / (l1 * l2);
        (s1 * t, s2 * t)
    }
}

/// The filled balloon, axisymmetric, from its bottom (z = 0) up to where the neck is tied.
pub struct Profile {
    /// (angle from the bottom on the uninflated sphere, r, z), z increasing.
    pub samples: Vec<(f64, f64, f64)>,
    pub neck_angle: f64,
    pub z_neck: f64,
    pub neck_r: f64,
    pub volume: f64,
    pub z_centroid: f64,
    /// The most the rubber is stretched round the balloon.
    pub stretch: f64,
}

impl Profile {
    pub fn z_top(&self) -> f64 {
        self.z_neck
    }

    /// Radius of the membrane at height z.
    pub fn r_at(&self, z: f64) -> f64 {
        let s = &self.samples;
        if z <= 0.0 || z >= self.z_neck {
            return 0.0;
        }
        let i = s.partition_point(|p| p.2 < z).clamp(1, s.len() - 1);
        let (a, b) = (s[i - 1], s[i]);
        a.1 + (b.1 - a.1) * (z - a.2) / (b.2 - a.2).max(1e-15)
    }

    /// Where the rubber at `angle` on the uninflated sphere is, as (r, z).
    pub fn place(&self, angle: f64) -> (f64, f64) {
        let s = &self.samples;
        let i = s.partition_point(|p| p.0 < angle).clamp(1, s.len() - 1);
        let (a, b) = (s[i - 1], s[i]);
        let f = ((angle - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
        (a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f)
    }
}

const SHOOT_STEPS: usize = 400;

/// Integrates the axisymmetric membrane equations from the bottom of the balloon to the neck,
/// over the angle on the uninflated sphere. The stretch and curvature at the bottom are the
/// unknowns. They set the pressure there, and hydrostatics sets it everywhere else. Returns
/// the samples and the volume inside, or None where the rubber closes up or goes slack
/// before it reaches the neck.
fn shoot(latex: &Latex, rest_r: f64, apex_stretch: f64, apex_curv: f64, rho_g: f64) -> Option<(Vec<(f64, f64, f64)>, f64)> {
    let neck_angle = PI - NECK.asin();
    let p_apex = 2.0 * latex.tension(apex_stretch, apex_stretch).0 * apex_curv;
    // d/ds of (r, z, slope angle, meridional stretch, volume), per length of the uninflated
    // meridian. The last two come from the balance of forces along the meridian.
    let deriv = |th: f64, y: [f64; 5]| -> Option<[f64; 5]> {
        let [r, z, psi, l1, _] = y;
        let rr = rest_r * th.sin();
        let l2 = r / rr;
        if !(r > 0.0 && l1 > 0.0) {
            return None;
        }
        let (t1, t2) = latex.tension(l1, l2);
        if t1 <= 0.0 {
            return None;
        }
        let e = 1e-6;
        let t1_l1 = (latex.tension(l1 * (1.0 + e), l2).0 - latex.tension(l1 * (1.0 - e), l2).0) / (2.0 * e * l1);
        let t1_l2 = (latex.tension(l1, l2 * (1.0 + e)).0 - latex.tension(l1, l2 * (1.0 - e)).0) / (2.0 * e * l2);
        let p = p_apex - rho_g * z;
        let dr = l1 * psi.cos();
        let dz = l1 * psi.sin();
        let dpsi = l1 * (p - t2 * psi.sin() / r) / t1;
        let dl2 = (dr - l2 * th.cos()) / rr;
        let dl1 = (dr * (t2 - t1) - r * t1_l2 * dl2) / (r * t1_l1);
        Some([dr, dz, dpsi, dl1, PI * r * r * dz].map(|d| d * rest_r))
    };
    let th0: f64 = 1e-3;
    let r = apex_stretch * rest_r * th0.sin();
    let mut y = [r, 0.5 * apex_curv * r * r, apex_curv * r, apex_stretch, 0.0];
    let h = (neck_angle - th0) / SHOOT_STEPS as f64;
    let mut th = th0;
    let mut samples = Vec::with_capacity(SHOOT_STEPS + 2);
    samples.push((0.0, 0.0, 0.0));
    samples.push((th, y[0], y[1]));
    let add = |y: [f64; 5], k: [f64; 5], s: f64| std::array::from_fn(|i| y[i] + s * k[i]);
    for _ in 0..SHOOT_STEPS {
        let k1 = deriv(th, y)?;
        let k2 = deriv(th + 0.5 * h, add(y, k1, 0.5 * h))?;
        let k3 = deriv(th + 0.5 * h, add(y, k2, 0.5 * h))?;
        let k4 = deriv(th + h, add(y, k3, h))?;
        y = std::array::from_fn(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        th += h;
        if y[0] <= 0.0 || y[2] > 3.5 {
            return None;
        }
        samples.push((th, y[0], y[1]));
    }
    Some((samples, y[4]))
}

/// Fills a balloon of uninflated radius `rest_r` with `volume` of water, from barely stretched
/// up, keeping to the shape it passes through the way it would filling from a tap. Rubber
/// has a limit point, past which it can take more than one shape at the same volume, and
/// this is the one filling reaches.
pub fn inflate(latex: &Latex, rest_r: f64, volume: f64, rho_g: f64) -> Profile {
    let neck_r = NECK * rest_r;
    // Unknowns (apex stretch, log apex curvature), and how far they are from closing on the
    // neck and holding log volume `target`.
    let miss = |x: [f64; 2], target: f64| -> Option<[f64; 2]> {
        let (s, v) = shoot(latex, rest_r, x[0], x[1].exp(), rho_g)?;
        Some([(s.last().unwrap().1 - neck_r) / neck_r, v.ln() - target])
    };

    // Barely inflated there's only one shape: bisect for the curvature that closes it.
    let la0 = 1.2;
    let (mut lo, mut hi) = ((1e-3 / rest_r).ln(), (1e3 / rest_r).ln());
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        match shoot(latex, rest_r, la0, mid.exp(), rho_g) {
            Some((s, _)) if s.last().unwrap().1 > neck_r => lo = mid,
            _ => hi = mid,
        }
    }
    let mut x = [la0, lo];
    let mut v = shoot(latex, rest_r, x[0], x[1].exp(), rho_g).expect("balloon doesn't inflate").1.ln();
    let mut prev = x;
    let n_inc = ((volume.ln() - v).abs() / 0.05).ceil().max(1.0) as usize;
    let inc = (volume.ln() - v) / n_inc as f64;
    for step in 0..n_inc {
        let target = v + inc;
        let mut y = if step > 0 { [2.0 * x[0] - prev[0], 2.0 * x[1] - prev[1]] } else { x };
        // Newton with a finite-difference Jacobian, halving steps that don't reduce the miss.
        let mut done = false;
        for _ in 0..30 {
            let Some(f) = miss(y, target) else { break };
            let norm = f[0].hypot(f[1]);
            if norm < 1e-9 {
                done = true;
                break;
            }
            let e = 1e-6;
            let (Some(fa), Some(fb)) = (miss([y[0] + e, y[1]], target), miss([y[0], y[1] + e], target)) else { break };
            let j = [[(fa[0] - f[0]) / e, (fb[0] - f[0]) / e], [(fa[1] - f[1]) / e, (fb[1] - f[1]) / e]];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            let d = [(-f[0] * j[1][1] + f[1] * j[0][1]) / det, (-f[1] * j[0][0] + f[0] * j[1][0]) / det];
            let mut a = 1.0;
            while a > 1e-4 {
                let c = [y[0] + a * d[0], y[1] + a * d[1]];
                if miss(c, target).is_some_and(|g| g[0].hypot(g[1]) < norm) {
                    y = c;
                    break;
                }
                a *= 0.5;
            }
        }
        assert!(done, "the balloon shape didn't converge at {:.3} L", target.exp() * 1e3);
        prev = x;
        x = y;
        v = target;
    }

    let (samples, volume) = shoot(latex, rest_r, x[0], x[1].exp(), rho_g).unwrap();
    let mut zmom = 0.0;
    for w in samples.windows(2) {
        let (a, b) = (w[0], w[1]);
        zmom += PI * 0.5 * (a.1 * a.1 + b.1 * b.1) * (b.2 - a.2) * 0.5 * (a.2 + b.2);
    }
    let stretch = samples.iter().map(|s| s.1 / (rest_r * s.0.sin()).max(1e-12)).skip(1).fold(0.0, f64::max);
    let last = *samples.last().unwrap();
    Profile {
        neck_angle: last.0,
        z_neck: last.2,
        neck_r,
        volume,
        z_centroid: zmom / volume,
        stretch,
        samples,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn icosphere(level: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..level {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut mid = |a: u32, b: u32, v: &mut Vec<Vec3>| -> u32 {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                v.push(((v[a as usize] + v[b as usize]) * 0.5).normalize());
                v.len() as u32 - 1
            })
        };
        let mut nf = Vec::with_capacity(f.len() * 4);
        for &[a, b, c] in &f {
            let ab = mid(a, b, &mut v);
            let bc = mid(b, c, &mut v);
            let ca = mid(c, a, &mut v);
            nf.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    (v, f)
}

/// Renumbers the mesh along a Morton curve through space, so neighbouring vertices and
/// triangles sit close in memory. The GPU reads each vertex's neighbours every substep.
fn spatial_order(v: &mut Vec<Vec3>, f: &mut [[u32; 3]]) {
    let spread = |x: u32| {
        let mut x = x & 0x3ff;
        x = (x | (x << 16)) & 0x030000ff;
        x = (x | (x << 8)) & 0x0300f00f;
        x = (x | (x << 4)) & 0x030c30c3;
        (x | (x << 2)) & 0x09249249
    };
    let key = |p: Vec3| {
        let q = ((p * 0.5 + 0.5) * 1023.0).as_uvec3();
        spread(q.x) | spread(q.y) << 1 | spread(q.z) << 2
    };
    let mut order: Vec<u32> = (0..v.len() as u32).collect();
    order.sort_by_key(|&i| key(v[i as usize]));
    let mut new_index = vec![0u32; v.len()];
    for (n, &old) in order.iter().enumerate() {
        new_index[old as usize] = n as u32;
    }
    *v = order.iter().map(|&i| v[i as usize]).collect();
    for t in f.iter_mut() {
        *t = t.map(|i| new_index[i as usize]);
    }
    f.sort_by_key(|t| key((v[t[0] as usize] + v[t[1] as usize] + v[t[2] as usize]) / 3.0));
}

pub struct Balloon {
    pub profile: Profile,
    pub latex: Latex,
    pub bottom_y: f64,
    pub center: Vec3,
    pub knot: Vec3,
    pub radius_max: f64,
    pub spacing: f64,
    pub particles: Vec<[f32; 4]>,
    pub mem_pos: Vec<[f32; 4]>,
    /// Rubber mass per vertex, 0 where it is held by the knot.
    pub mem_mass: Vec<f32>,
    pub tris: Vec<GpuTri>,
    pub vtri_off: Vec<u32>,
    pub vtri: Vec<u32>,
    pub stretch: f64,
    pub thickness0: f64,
}

pub fn build(p: &SimParams) -> Balloon {
    let d = p.balloon_diameter_cm as f64 * 0.01;
    let volume = PI / 6.0 * d * d * d;
    let t0 = p.rubber_thickness_mm as f64 * 1e-3;
    let latex = Latex::new(p.rubber_modulus_mpa as f64 * 1e6, t0);
    let rest_r = p.rubber_rest_diameter_cm as f64 * 0.005;
    let profile = inflate(&latex, rest_r, volume, RHO_WATER * p.gravity.max(0.0) as f64);
    let bottom_y = p.balloon_height as f64 - profile.z_centroid;
    let radius_max = profile.samples.iter().map(|q| q.1).fold(0.0, f64::max);
    let center = Vec3::new(0.0, p.balloon_height, 0.0);
    let knot = Vec3::new(0.0, (bottom_y + profile.z_neck + 0.5 * profile.neck_r) as f32, 0.0);

    // The mesh is laid out on the uninflated sphere, bottom at -y, and each point goes where
    // the inflation took that bit of rubber. Past the neck is the knot, which holds still.
    let (mut sphere, mut faces) = icosphere(6);
    spatial_order(&mut sphere, &mut faces);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let edge_angle = 1.1 / (1u32 << 6) as f32;
    let mut rest: Vec<DVec3> = Vec::with_capacity(sphere.len());
    let mut pos: Vec<Vec3> = Vec::with_capacity(sphere.len());
    let mut held = Vec::with_capacity(sphere.len());
    for &s in &sphere {
        // Jitter along the sphere so tears don't follow the icosahedral grid.
        let t1 = s.any_orthonormal_vector();
        let t2 = s.cross(t1);
        let j = (t1 * (rng.next() as f32 - 0.5) + t2 * (rng.next() as f32 - 0.5)) * edge_angle * 0.4;
        let dir = (s + j).normalize().as_dvec3();
        rest.push(dir * rest_r);
        let angle = (-dir.y).clamp(-1.0, 1.0).acos();
        let az = dir.z.atan2(dir.x);
        let (r, z) = if angle <= profile.neck_angle {
            profile.place(angle)
        } else {
            let f = (angle - profile.neck_angle) / (PI - profile.neck_angle);
            (profile.neck_r * (1.0 - f), profile.z_neck + 0.5 * profile.neck_r * f)
        };
        held.push(angle > profile.neck_angle);
        pos.push(DVec3::new(r * az.cos(), bottom_y + z, r * az.sin()).as_vec3());
    }

    let mut vmass = vec![0.0f64; pos.len()];
    let mut tris: Vec<GpuTri> = Vec::with_capacity(faces.len());
    for f in faces.iter_mut() {
        let [a, b, c] = f.map(|i| rest[i as usize]);
        if (b - a).cross(c - a).dot(a + b + c) < 0.0 {
            f.swap(1, 2);
        }
        let [a, b, c] = f.map(|i| rest[i as usize]);
        let (e1, e2) = (b - a, c - a);
        let u = e1.normalize();
        let w = e1.cross(e2).normalize().cross(u);
        let (m00, m01, m10, m11) = (e1.dot(u), e2.dot(u), e1.dot(w), e2.dot(w));
        let det = m00 * m11 - m01 * m10;
        let area = 0.5 * det.abs();
        for &v in f.iter() {
            vmass[v as usize] += RHO_RUBBER * t0 * area / 3.0;
        }
        let dm_inv = [m11 / det, -m01 / det, -m10 / det, m00 / det].map(|x| x as f32);
        tris.push(GpuTri { v: *f, rest_area: area as f32, dm_inv });
    }
    let mem_mass = vmass.iter().zip(&held).map(|(&m, &h)| if h { 0.0 } else { m as f32 }).collect();
    let mem_pos = pos.iter().map(|q| [q.x, q.y, q.z, 0.0]).collect();

    let mut vt: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
    for (ti, t) in tris.iter().enumerate() {
        for &v in &t.v {
            vt[v as usize].push(ti as u32);
        }
    }
    let mut vtri_off = vec![0u32];
    let mut vtri = Vec::new();
    for l in &vt {
        vtri.extend_from_slice(l);
        vtri_off.push(vtri.len() as u32);
    }

    // Water particles fill the membrane on a jittered lattice.
    let spacing = d / p.resolution.max(16) as f64;
    let mut particles = Vec::new();
    let n = (2.0 * radius_max / spacing).ceil() as i32 + 2;
    let nz = (profile.z_neck / spacing).ceil() as i32 + 1;
    for iy in 0..nz {
        let zl = (iy as f64 + 0.5) * spacing;
        if zl > profile.z_neck - spacing {
            break;
        }
        let rmax = profile.r_at(zl) - 0.5 * spacing;
        if rmax <= 0.0 {
            continue;
        }
        for ix in -n / 2..=n / 2 {
            for iz in -n / 2..=n / 2 {
                let x = (ix as f64 + 0.5 * (iy % 2) as f64) * spacing;
                let z = (iz as f64 + 0.5 * (iy % 2) as f64) * spacing;
                let jx = (rng.next() - 0.5) * 0.3 * spacing;
                let jy = (rng.next() - 0.5) * 0.3 * spacing;
                let jz = (rng.next() - 0.5) * 0.3 * spacing;
                if (x * x + z * z).sqrt() < rmax {
                    particles.push([(x + jx) as f32, (bottom_y + zl + jy) as f32, (z + jz) as f32, 1.0]);
                }
            }
        }
    }

    Balloon {
        stretch: profile.stretch,
        profile,
        latex,
        bottom_y,
        center,
        knot,
        radius_max,
        spacing,
        particles,
        mem_pos,
        mem_mass,
        tris,
        vtri_off,
        vtri,
        thickness0: t0,
    }
}

impl Balloon {
    /// Where a straight line through `aim` along `dir` first meets the undisturbed balloon.
    pub fn entry(&self, aim: Vec3, dir: Vec3) -> Option<Vec3> {
        let inside = |s: f64| {
            let q = aim + dir * s as f32;
            (q.x as f64).hypot(q.z as f64) < self.profile.r_at(q.y as f64 - self.bottom_y)
        };
        let reach = 1.5 * (self.radius_max + self.profile.z_top());
        let steps = 8000;
        (0..=steps).map(|i| -reach + 2.0 * reach * i as f64 / steps as f64).find(|&s| inside(s)).map(|s| aim + dir * s as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflation_holds_the_volume_and_hangs_pear_shaped() {
        let latex = Latex::new(0.42e6, 0.25e-3);
        let d = 0.16;
        let v = PI / 6.0 * d * d * d;
        let p = inflate(&latex, 0.03, v, RHO_WATER * 9.81);
        let rmax = p.samples.iter().map(|q| q.1).fold(0.0, f64::max);
        println!("volume {:.4} L (target {:.4}), height {:.3}, width {:.3}, centroid {:.3}, stretch {:.2}",
            p.volume * 1e3, v * 1e3, p.z_neck, 2.0 * rmax, p.z_centroid, p.stretch);
        assert!((p.volume - v).abs() / v < 1e-6);
        assert!(p.z_centroid < 0.5 * p.z_neck);
        assert!((p.samples.last().unwrap().1 - p.neck_r).abs() < 1e-6 * p.neck_r);
    }

    #[test]
    fn weightless_balloon_is_round() {
        let latex = Latex::new(0.42e6, 0.25e-3);
        let d = 0.16;
        let p = inflate(&latex, 0.03, PI / 6.0 * d * d * d, 0.0);
        let rmax = p.samples.iter().map(|q| q.1).fold(0.0, f64::max);
        // The neck pulls the top in a little, so compare the lower half with a sphere.
        let r = p.r_at(rmax);
        assert!((r - rmax).abs() / rmax < 0.02, "radius at its own height {r:.4} vs widest {rmax:.4}");
    }

    #[test]
    fn scene_builds() {
        let b = build(&SimParams::default());
        println!("particles {}, verts {}, tris {}", b.particles.len(), b.mem_pos.len(), b.tris.len());
        assert!(b.entry(b.center, Vec3::X).is_some());
    }
}
