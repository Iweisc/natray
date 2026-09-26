//! CPU-side scene construction: the hanging balloon's equilibrium shape, the latex
//! membrane mesh and the water particles. The bullets are in flight.rs.

use crate::params::SimParams;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;

pub const RHO_WATER: f64 = 1000.0;
pub const RHO_RUBBER: f64 = 950.0;
pub const NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuTri {
    pub v: [u32; 3],
    pub rest_area: f32,
    pub e: [u32; 3],
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuEdge {
    pub v0: u32,
    pub v1: u32,
    pub t0: u32,
    pub t1: u32,
    pub rest: f32,
    pub thresh: f32,
    pub k: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVInfo {
    pub start: u32,
    pub count: u32,
    pub inv_mass: f32,
    pub mass: f32,
}

/// Axisymmetric balloon profile, from the bottom apex (z = 0) to the knot.
pub struct Profile {
    pub rz: Vec<(f64, f64)>,
    pub z_neck: f64,
    pub neck_r: f64,
    pub volume: f64,
    pub z_centroid: f64,
}

impl Profile {
    pub fn z_top(&self) -> f64 {
        self.rz.last().unwrap().1
    }

    /// Radius of the membrane at height z (z is monotonic along the profile).
    pub fn r_at(&self, z: f64) -> f64 {
        if z <= 0.0 || z >= self.z_top() {
            return 0.0;
        }
        let i = self.rz.partition_point(|p| p.1 < z).clamp(1, self.rz.len() - 1);
        let (r0, z0) = self.rz[i - 1];
        let (r1, z1) = self.rz[i];
        let f = if z1 > z0 { (z - z0) / (z1 - z0) } else { 0.0 };
        r0 + (r1 - r0) * f
    }
}

/// Integrates the Young-Laplace equation for a membrane with uniform tension holding
/// water (the same ODE as a pendant drop), then tapers the upper part into the knotted neck.
fn build_profile(kappa0: f64, tension: f64, rho_g: f64, neck_r: f64) -> Profile {
    let dp0 = 2.0 * tension * kappa0;
    let deriv = |r: f64, z: f64, psi: f64| -> (f64, f64, f64) {
        (psi.cos(), psi.sin(), (dp0 - rho_g * z) / tension - psi.sin() / r.max(1e-9))
    };
    let s0 = 1e-4;
    let (mut r, mut z, mut psi) = (s0, 0.5 * kappa0 * s0 * s0, kappa0 * s0);
    let mut pts = vec![(0.0, 0.0), (r, z)];
    let ds = 2.5e-4;
    let psi_stop = 2.15;
    for _ in 0..40000 {
        let k1 = deriv(r, z, psi);
        let k2 = deriv(r + 0.5 * ds * k1.0, z + 0.5 * ds * k1.1, psi + 0.5 * ds * k1.2);
        let k3 = deriv(r + 0.5 * ds * k2.0, z + 0.5 * ds * k2.1, psi + 0.5 * ds * k2.2);
        let k4 = deriv(r + ds * k3.0, z + ds * k3.1, psi + ds * k3.2);
        r += ds / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        z += ds / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        psi += ds / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
        pts.push((r, z));
        let turning = deriv(r, z, psi).2;
        if psi >= psi_stop || (psi > 1.6 && r < neck_r * 2.0) || (psi > 1.8 && turning < 0.15 * kappa0) {
            break;
        }
    }

    // Taper from the shoulder to the neck: a cubic Hermite curve. The neck leaves almost
    // vertically, which gives the teardrop look of a water balloon hanging from its knot.
    let (rc, zc) = (r, z);
    let tc = (psi.cos(), psi.sin());
    let psi_n: f64 = 1.72;
    let tn = (psi_n.cos(), psi_n.sin());
    let rn = neck_r;
    let zn = zc + 1.05 * (rc - rn).max(0.0) + 0.004;
    let chord = ((rc - rn).powi(2) + (zn - zc).powi(2)).sqrt();
    let n_taper = 80;
    for i in 1..=n_taper {
        let u = i as f64 / n_taper as f64;
        let h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0;
        let h10 = u * u * u - 2.0 * u * u + u;
        let h01 = -2.0 * u * u * u + 3.0 * u * u;
        let h11 = u * u * u - u * u;
        let pr = h00 * rc + h10 * chord * tc.0 + h01 * rn + h11 * chord * tn.0;
        let pz = h00 * zc + h10 * chord * tc.1 + h01 * zn + h11 * chord * tn.1;
        pts.push((pr.max(0.0), pz));
    }
    // Rounded cap under the knot.
    let n_cap = 12;
    for i in 1..=n_cap {
        let a = i as f64 / n_cap as f64 * std::f64::consts::FRAC_PI_2;
        pts.push((rn * a.cos(), zn + 0.5 * rn * a.sin()));
    }
    // z must be monotonic for r_at(); drop any sample that turns back down.
    let mut rz: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for p in pts {
        if rz.last().map_or(true, |l: &(f64, f64)| p.1 > l.1) {
            rz.push(p);
        }
    }
    let mut volume = 0.0;
    let mut zmom = 0.0;
    for w in rz.windows(2) {
        let (r0, z0) = w[0];
        let (r1, z1) = w[1];
        let a = std::f64::consts::PI * 0.5 * (r0 * r0 + r1 * r1);
        volume += a * (z1 - z0);
        zmom += a * (z1 - z0) * 0.5 * (z0 + z1);
    }
    Profile { rz, z_neck: zn, neck_r: rn, volume, z_centroid: zmom / volume.max(1e-12) }
}

pub fn solve_profile(volume: f64, tension: f64, gravity: f64) -> Profile {
    let d = (6.0 * volume / std::f64::consts::PI).cbrt();
    let rho_g = RHO_WATER * gravity.max(0.0);
    let neck_r = (0.045 * d).clamp(0.004, 0.009);
    let (mut lo, mut hi) = (0.4 / d, 12.0 / d);
    for _ in 0..60 {
        let mid = (lo * hi).sqrt();
        if build_profile(mid, tension, rho_g, neck_r).volume > volume {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    build_profile((lo * hi).sqrt(), tension, rho_g, neck_r)
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

pub struct Balloon {
    pub profile: Profile,
    pub bottom_y: f64,
    pub center: Vec3,
    pub knot: Vec3,
    pub radius_max: f64,
    pub spacing: f64,
    pub particles: Vec<[f32; 4]>,
    pub mem_pos: Vec<[f32; 4]>,
    pub tris: Vec<GpuTri>,
    pub edges: Vec<GpuEdge>,
    pub vinfo: Vec<GpuVInfo>,
    pub adj: Vec<[u32; 2]>,
    pub vtri_off: Vec<u32>,
    pub vtri: Vec<u32>,
    pub dt_mem_max: f64,
    pub stretch: f64,
    pub thickness0: f64,
}

pub fn build(p: &SimParams) -> Balloon {
    let d = p.balloon_diameter_cm as f64 * 0.01;
    let volume = std::f64::consts::PI / 6.0 * d * d * d;
    let tension = p.rubber_tension as f64;
    let profile = solve_profile(volume, tension, p.gravity as f64);
    let bottom_y = p.balloon_height as f64 - profile.z_centroid;
    let radius_max = profile.rz.iter().map(|q| q.0).fold(0.0, f64::max);
    let center = Vec3::new(0.0, p.balloon_height, 0.0);
    let knot = Vec3::new(0.0, (bottom_y + profile.z_top()) as f32 + 0.002, 0.0);

    // Equal-area map from the unit sphere onto the balloon surface keeps triangle sizes even.
    let mut cum_area = vec![0.0f64];
    for w in profile.rz.windows(2) {
        let (r0, z0) = w[0];
        let (r1, z1) = w[1];
        let seg = ((r1 - r0).powi(2) + (z1 - z0).powi(2)).sqrt();
        cum_area.push(cum_area.last().unwrap() + std::f64::consts::PI * (r0 + r1) * seg);
    }
    let total_area = *cum_area.last().unwrap();
    let map_frac = |frac: f64| -> (f64, f64) {
        let target = frac * total_area;
        let i = cum_area.partition_point(|&a| a < target).clamp(1, cum_area.len() - 1);
        let f = (target - cum_area[i - 1]) / (cum_area[i] - cum_area[i - 1]).max(1e-15);
        let (r0, z0) = profile.rz[i - 1];
        let (r1, z1) = profile.rz[i];
        (r0 + (r1 - r0) * f, z0 + (z1 - z0) * f)
    };

    let (sphere, faces) = icosphere(6);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let edge_angle = 1.1 / (1u32 << 6) as f32;
    let mut pos: Vec<Vec3> = Vec::with_capacity(sphere.len());
    for &s in &sphere {
        // Jitter along the sphere so tears don't follow the icosahedral grid.
        let t1 = s.any_orthonormal_vector();
        let t2 = s.cross(t1);
        let j = (t1 * (rng.next() as f32 - 0.5) + t2 * (rng.next() as f32 - 0.5)) * edge_angle * 0.4;
        let dir = (s + j).normalize();
        let frac = (1.0 + dir.y as f64) * 0.5; // area fraction of the cap below this point
        let (r, z) = map_frac(frac);
        let az = dir.z.atan2(dir.x) as f64;
        pos.push(Vec3::new((r * az.cos()) as f32, (bottom_y + z) as f32, (r * az.sin()) as f32));
    }

    let stretch = p.rubber_stretch as f64;
    let t0 = p.rubber_thickness_mm as f64 * 1e-3;
    let areal_density = RHO_RUBBER * t0 / (stretch * stretch);

    let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut edges: Vec<GpuEdge> = Vec::new();
    let mut tris: Vec<GpuTri> = Vec::with_capacity(faces.len());
    let mut vmass = vec![0.0f64; pos.len()];
    for (ti, &[a, b, c]) in faces.iter().enumerate() {
        let pa = pos[a as usize];
        let pb = pos[b as usize];
        let pc = pos[c as usize];
        let area = 0.5 * (pb - pa).cross(pc - pa).length() as f64;
        for &v in &[a, b, c] {
            vmass[v as usize] += areal_density * area / 3.0;
        }
        let mut e = [0u32; 3];
        for (k, &(u, w)) in [(a, b), (b, c), (c, a)].iter().enumerate() {
            let key = (u.min(w), u.max(w));
            let id = *edge_map.entry(key).or_insert_with(|| {
                edges.push(GpuEdge {
                    v0: key.0,
                    v1: key.1,
                    t0: ti as u32,
                    t1: NONE,
                    rest: 0.0,
                    thresh: 0.0,
                    k: 0.0,
                    _pad: 0.0,
                });
                edges.len() as u32 - 1
            });
            if edges[id as usize].t0 != ti as u32 {
                edges[id as usize].t1 = ti as u32;
            }
            e[k] = id;
        }
        tris.push(GpuTri { v: [a, b, c], rest_area: (area / (stretch * stretch)) as f32, e, _pad: 0 });
    }

    let neck_z = (bottom_y + profile.z_neck) as f32 - 0.004;
    let strength = p.rubber_strength as f64;

    // Uniform membrane tension T on a triangle mesh is the area gradient times T, which
    // puts a force of T * (cot a + cot b) / 2 * L along each edge (cotangent weights).
    // Using those as the pre-tension keeps the irregular mesh in equilibrium at t = 0.
    let mut cot_w = vec![0.0f64; edges.len()];
    for t in &tris {
        let [a, b, c] = t.v;
        let pts = [pos[a as usize], pos[b as usize], pos[c as usize]];
        for k in 0..3 {
            let p0 = pts[k];
            let p1 = pts[(k + 1) % 3];
            let p2 = pts[(k + 2) % 3];
            // Angle at p0 is opposite the edge p1-p2, which is t.e[(k + 1) % 3].
            let u = (p1 - p0).as_dvec3();
            let v = (p2 - p0).as_dvec3();
            let cot = u.dot(v) / u.cross(v).length().max(1e-15);
            cot_w[t.e[(k + 1) % 3] as usize] += 0.5 * cot;
        }
    }
    let mut stiff_sum = vec![0.0f64; pos.len()];
    for (i, e) in edges.iter_mut().enumerate() {
        let len = (pos[e.v0 as usize] - pos[e.v1 as usize]).length() as f64;
        let w = cot_w[i].clamp(0.15, 1.5);
        let k = tension * w * len / (stretch - 1.0).max(0.05);
        e.rest = (len / stretch) as f32;
        e.k = k as f32;
        let u = rng.next();
        // Weibull-like scatter in rubber strength, so cracks wander and branch.
        e.thresh = (stretch * strength * (1.0 + 0.12 * (-(1.0 - u).ln()).powf(0.5) - 0.06)) as f32;
        // The neck is barely inflated, so its rubber is thick and cracks stop short of it.
        // The strips then stay tied to the knot and snap back around it.
        let h = ((pos[e.v0 as usize].y + pos[e.v1 as usize].y) as f64 * 0.5 - bottom_y) / profile.z_neck;
        let t = ((h - 0.8) / 0.17).clamp(0.0, 1.0);
        e.thresh *= (1.0 + 3.0 * t * t * (3.0 - 2.0 * t)) as f32;
        let kk = k / (len / stretch);
        stiff_sum[e.v0 as usize] += kk;
        stiff_sum[e.v1 as usize] += kk;
    }
    // Lumped masses of sliver-shaped vertices are tiny and would dictate the time step;
    // topping them up to a fraction of the median costs a negligible amount of extra mass.
    let mut sorted = vmass.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let floor_mass = 0.7 * sorted[sorted.len() / 2];
    for m in vmass.iter_mut() {
        *m = m.max(floor_mass);
    }

    let mut adj_lists: Vec<Vec<[u32; 2]>> = vec![Vec::new(); pos.len()];
    for (i, e) in edges.iter().enumerate() {
        adj_lists[e.v0 as usize].push([e.v1, i as u32]);
        adj_lists[e.v1 as usize].push([e.v0, i as u32]);
    }
    let mut adj = Vec::new();
    let mut vinfo = Vec::with_capacity(pos.len());
    let mut dt_mem_max = f64::MAX;
    let mut mem_pos = Vec::with_capacity(pos.len());
    for (i, list) in adj_lists.iter().enumerate() {
        let pinned = pos[i].y > neck_z;
        let m = vmass[i].max(1e-9);
        vinfo.push(GpuVInfo {
            start: adj.len() as u32,
            count: list.len() as u32,
            inv_mass: if pinned { 0.0 } else { (1.0 / m) as f32 },
            mass: m as f32,
        });
        adj.extend_from_slice(list);
        if !pinned {
            dt_mem_max = dt_mem_max.min((m / stiff_sum[i]).sqrt());
        }
        mem_pos.push([pos[i].x, pos[i].y, pos[i].z, if pinned { 0.0 } else { 1.0 }]);
    }
    dt_mem_max *= 0.45;

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
        profile,
        bottom_y,
        center,
        knot,
        radius_max,
        spacing,
        particles,
        mem_pos,
        tris,
        edges,
        vinfo,
        adj,
        vtri_off,
        vtri,
        dt_mem_max,
        stretch,
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
    fn profile_matches_volume_and_hangs_pear_shaped() {
        let d = 0.16;
        let v = std::f64::consts::PI / 6.0 * d * d * d;
        let p = solve_profile(v, 140.0, 9.81);
        let rmax = p.rz.iter().map(|q| q.0).fold(0.0, f64::max);
        println!("volume {:.4} L (target {:.4}), height {:.3}, width {:.3}, centroid {:.3}, neck {:.3}",
            p.volume * 1e3, v * 1e3, p.z_top(), 2.0 * rmax, p.z_centroid, p.z_neck);
        assert!((p.volume - v).abs() / v < 0.01);
        assert!(p.z_centroid < 0.5 * p.z_top());
    }

    #[test]
    fn scene_builds() {
        let b = build(&SimParams::default());
        println!("particles {}, verts {}, tris {}, dt_mem {:.2e}", b.particles.len(), b.mem_pos.len(), b.tris.len(), b.dt_mem_max);
        assert!(b.entry(b.center, Vec3::X).is_some());
    }
}
