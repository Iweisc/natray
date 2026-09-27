//! Rigid-body flight of a spinning bullet through air and water.
//!
//! The bullet is a body of revolution cut into slices along its axis. Every slice in a fluid
//! takes cross-flow drag, Magnus lift and skin friction, and the leading end takes the drag
//! and, for a pointed nose, the lift that tips it over. In air that lift is scaled so the
//! gyroscopic stability matches Miller's twist rule, and the spin keeps the bullet pointed.
//! In water the nose opens a vapour cavity and the body behind it stays dry until the yaw
//! swings the tail out through the cavity wall, where the water slaps it back. The
//! overturning moment in water is hundreds of times what any spin can hold, so every bullet
//! yaws there, and a long pointed one tumbles within a balloon's width and carries on base
//! first.

use crate::balloon::RHO_WATER;
use crate::cache::T_END;
use crate::lip::{cavity, P_ATM};
use crate::params::{Metal, Nose, Shot};
use glam::{DVec3, Vec3};
use std::f64::consts::{PI, TAU};

const RHO_AIR: f64 = 1.2;
const SLICES: usize = 24;
/// Samples of the path table per shot, evenly spaced along the chord through the water.
pub const NPATH: usize = 256;
/// Cross-flow drag of a cylinder: free-streamline (cavitating) in water, subcritical in air.
const CDC_WATER: f64 = 0.5;
const CDC_AIR: f64 = 1.2;
/// A flat base leading, after a tumble.
const CD_BASE_WATER: f64 = 0.82;
const CD_BASE_AIR: f64 = 1.17;
/// Turbulent skin friction.
const CF: f64 = 0.003;
/// Pitch damping in air, C_Mq + C_Ma', typical of small-arms bullets.
const CMQ_AIR: f64 = -10.0;
/// A bullet slower than this inside water has stopped there.
const V_STOP: f64 = 3.0;
/// The water phase takes over this far before the bullet reaches the balloon.
pub const LOOKAHEAD: f64 = 0.05;
/// Poses are kept while the bullet is this close to the balloon.
const NEAR: f64 = 0.8;

fn body_share(nose: Nose) -> f64 {
    match nose {
        Nose::Round => 0.55,
        Nose::Spitzer => 0.42,
        _ => 1.0,
    }
}

/// Radius of the body `u` from the base. Matches bullet_sdf in scene.wgsl.
pub fn profile(nose: Nose, r: f64, len: f64, u: f64) -> f64 {
    if nose == Nose::Sphere {
        return (r * r - (u - r).powi(2)).max(0.0).sqrt();
    }
    let body = body_share(nose) * len;
    if u <= body {
        return r;
    }
    let (x, nl) = (u - body, len - body);
    if nose == Nose::Spitzer {
        // Tangent ogive.
        let ro = (r * r + nl * nl) / (2.0 * r);
        ((ro * ro - x * x).max(0.0).sqrt() - (ro - r)).max(0.0)
    } else {
        r * (1.0 - (x / nl).powi(2)).max(0.0).sqrt()
    }
}

/// Miller's twist rule: gyroscopic stability of a bullet of mass `m` (kg) and diameter `d`
/// (m), `l` calibres long, in a barrel with one turn per `t` calibres, at speed `v` (m/s).
fn miller_sg(m: f64, d: f64, l: f64, t: f64, v: f64) -> f64 {
    let grains = m / 64.79891e-6;
    let d_in = d / 0.0254;
    let fps = v / 0.3048;
    30.0 * grains / (t * t * d_in.powi(3) * l * (1.0 + l * l)) * (fps / 2800.0).cbrt()
}

/// Every `u` here is measured along the axis from the base.
pub struct Body {
    pub nose: Nose,
    pub metal: Metal,
    pub radius: f64,
    pub length: f64,
    pub mass: f64,
    pub u_cg: f64,
    pub ix: f64,
    pub iy: f64,
    /// Centre, length and radius of each slice.
    slices: [(f64, f64, f64); SLICES],
    /// Silhouette area seen from the side.
    side_area: f64,
    /// Widest point of the nose, where the cavity leaves it.
    u_shoulder: f64,
    /// Where the nose's lift acts (slender-body theory), and where its drag does in water.
    u_cp: f64,
    u_drag: f64,
    cd_water: f64,
    cd_air: f64,
    /// Share of the slender-body lift the nose carries, in water and in air.
    kappa_water: f64,
    kappa_air: f64,
}

impl Body {
    pub fn new(s: &Shot) -> Self {
        let nose = s.bullet.nose;
        let r = 0.5 * s.caliber_mm as f64 * 1e-3;
        let mass = s.mass_g as f64 * 1e-3;
        // Jacketed bullets have a lead core, so they are only a little lighter than lead.
        let density = match s.bullet.metal {
            Metal::Lead => 11340.0,
            Metal::Copper => 10500.0,
            Metal::Steel => 7850.0,
        };
        // The share of the enclosing cylinder the shape takes up.
        let fill = match nose {
            Nose::Round => 0.85,
            Nose::Spitzer => 0.72,
            Nose::Flat => 0.95,
            Nose::Sphere => 1.0,
        };
        let length = if nose == Nose::Sphere { 2.0 * r } else { mass / density / (PI * r * r) / fill };

        // Mass properties of the actual shape, scaled to the given mass.
        let fine = 2000;
        let du = length / fine as f64;
        let rad: Vec<(f64, f64)> = (0..fine).map(|i| (i as f64 + 0.5) * du).map(|u| (u, profile(nose, r, length, u))).collect();
        let vol: f64 = rad.iter().map(|q| PI * q.1 * q.1 * du).sum();
        let u_cg = rad.iter().map(|q| PI * q.1 * q.1 * du * q.0).sum::<f64>() / vol;
        let rho = mass / vol;
        let ix = rad.iter().map(|q| rho * PI * q.1.powi(4) / 2.0 * du).sum();
        let iy = rad.iter().map(|q| rho * PI * q.1 * q.1 * du * (q.1 * q.1 / 4.0 + (q.0 - u_cg).powi(2))).sum();

        let body = body_share(nose) * length;
        let v_nose: f64 = rad.iter().filter(|q| q.0 > body).map(|q| PI * q.1 * q.1 * du).sum();
        let u_cp = if nose == Nose::Sphere { r } else { body + v_nose / (PI * r * r) };
        let (u_shoulder, u_drag) = match nose {
            Nose::Spitzer => (body, u_cp),
            // A blunt nose takes its drag like a sphere would, through the centre of its
            // tip's curvature.
            Nose::Round => (body, length - r * r / (length - body)),
            Nose::Flat => (length, length),
            Nose::Sphere => (r, r),
        };
        // Cavitator drag at zero cavitation number in water. In air the usual values for
        // these shapes around the speed of sound.
        let cd_water = match nose {
            Nose::Round | Nose::Sphere => 0.30,
            Nose::Spitzer => 0.20,
            Nose::Flat => 0.82,
        };
        let cd_air = match nose {
            Nose::Round | Nose::Spitzer => 0.30,
            Nose::Flat => 0.9,
            Nose::Sphere => 0.47,
        };
        let kappa_water = if nose == Nose::Spitzer { 1.0 } else { 0.0 };
        // The rule's stability goes with the spin squared, like the gyroscopic term does, so
        // any twist gives the same moment.
        let kappa_air = if nose == Nose::Sphere {
            0.0
        } else {
            let (v, d, t) = ((s.speed as f64).max(1.0), 2.0 * r, 30.0);
            let p = TAU * v / (t * d);
            let m_alpha = ix * ix * p * p / (4.0 * iy * miller_sg(mass, d, length / d, t, v));
            m_alpha / (RHO_AIR * PI * r * r * v * v * (u_cp - u_cg))
        };
        let slices = std::array::from_fn(|j| {
            let du = length / SLICES as f64;
            let u = (j as f64 + 0.5) * du;
            (u, du, profile(nose, r, length, u))
        });
        let side_area = rad.iter().map(|q| 2.0 * q.1 * du).sum();
        Self {
            nose,
            metal: s.bullet.metal,
            radius: r,
            length,
            mass,
            u_cg,
            ix,
            iy,
            slices,
            side_area,
            u_shoulder,
            u_cp,
            u_drag,
            cd_water,
            cd_air,
            kappa_water,
            kappa_air,
        }
    }
}

/// Water fill (0 air, 1 water) and velocity on a grid, splatted from the MPM particles and
/// sampled trilinearly. Fill 0.5 is the water surface.
///
/// Splatting and sampling each blur the surface over a grid cell, so the raw fill ramps up
/// over two centimetres. It is steepened back to about a particle spacing.
pub struct Field {
    origin: DVec3,
    h: f64,
    dims: [usize; 3],
    mass: Vec<f32>,
    mom: Vec<[f32; 3]>,
    full: f64,
}

impl Field {
    fn empty() -> Self {
        Self { origin: DVec3::ZERO, h: 1.0, dims: [0; 3], mass: Vec::new(), mom: Vec::new(), full: 1.0 }
    }

    /// `parts` are the live particles as (position, velocity).
    pub fn new(parts: &[(Vec3, Vec3)], spacing: f64) -> Self {
        if parts.is_empty() {
            return Self::empty();
        }
        let h = 2.0 * spacing;
        let (lo, hi) = parts.iter().fold((DVec3::MAX, DVec3::MIN), |(lo, hi), p| (lo.min(p.0.as_dvec3()), hi.max(p.0.as_dvec3())));
        let origin = lo - DVec3::splat(3.0 * h);
        let d = ((hi - origin) / h).ceil();
        let dims = [d.x as usize + 4, d.y as usize + 4, d.z as usize + 4];
        let mut f = Self { origin, h, dims, mass: vec![0.0; dims[0] * dims[1] * dims[2]], mom: vec![[0.0; 3]; dims[0] * dims[1] * dims[2]], full: (h / spacing).powi(3) };
        for (p, v) in parts {
            let g = (p.as_dvec3() - origin) / h;
            let b = g.floor();
            let fr = g - b;
            for c in 0..8 {
                let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
                let w = [fr.x, fr.y, fr.z].iter().zip(o).map(|(&f, o)| if o == 1 { f } else { 1.0 - f }).product::<f64>() as f32;
                let i = f.index(b.x as usize + o[0], b.y as usize + o[1], b.z as usize + o[2]);
                f.mass[i] += w;
                for k in 0..3 {
                    f.mom[i][k] += w * v[k];
                }
            }
        }
        f
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims[1] + j) * self.dims[2] + k
    }

    fn at(&self, p: DVec3) -> (f64, DVec3) {
        let g = (p - self.origin) / self.h;
        let b = g.floor();
        if b.min_element() < 0.0 || b.x as usize + 1 >= self.dims[0] || b.y as usize + 1 >= self.dims[1] || b.z as usize + 1 >= self.dims[2] {
            return (0.0, DVec3::ZERO);
        }
        let fr = g - b;
        let (mut m, mut mom) = (0.0, DVec3::ZERO);
        for c in 0..8 {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let w = [fr.x, fr.y, fr.z].iter().zip(o).map(|(&f, o)| if o == 1 { f } else { 1.0 - f }).product::<f64>();
            let i = self.index(b.x as usize + o[0], b.y as usize + o[1], b.z as usize + o[2]);
            m += w * self.mass[i] as f64;
            mom += w * DVec3::from(self.mom[i].map(|x| x as f64));
        }
        let fill = ((m / self.full - 0.5) * 3.0 + 0.5).clamp(0.0, 1.0);
        (fill, if m > 1e-6 { mom / m } else { DVec3::ZERO })
    }
}

#[derive(Clone, Copy)]
struct State {
    x: DVec3,
    u: DVec3,
    /// Unit axis towards the nose.
    n: DVec3,
    /// Angular momentum.
    h: DVec3,
    roll: f64,
}

impl State {
    fn add(&self, d: &State, k: f64) -> State {
        State { x: self.x + d.x * k, u: self.u + d.u * k, n: self.n + d.n * k, h: self.h + d.h * k, roll: self.roll + d.roll * k }
    }

    fn lerp(&self, o: &State, k: f64) -> State {
        let mut s = self.add(&o.add(self, -1.0), k);
        s.n = s.n.normalize();
        s
    }

    fn yaw(&self) -> f64 {
        self.n.dot(self.u.normalize_or_zero()).clamp(-1.0, 1.0).acos()
    }
}

#[derive(Clone, Copy, Default)]
struct Loads {
    force: DVec3,
    torque: DVec3,
    /// Force and spin torque the water takes from the bullet.
    on_water: DVec3,
    spin_on_water: f64,
    /// Radius of a disk with the silhouette's area, seen along the path.
    r_eff: f64,
}

#[derive(Clone, Copy)]
pub struct Pose {
    pub t: f64,
    pub x: DVec3,
    pub u: DVec3,
    pub n: DVec3,
    pub roll: f64,
    pub spin: f64,
    loads: Loads,
}

impl Pose {
    fn lerp(&self, o: &Pose, k: f64) -> Pose {
        let l = |a: DVec3, b: DVec3| a + (b - a) * k;
        let s = |a: f64, b: f64| a + (b - a) * k;
        let (a, b) = (&self.loads, &o.loads);
        Pose {
            t: s(self.t, o.t),
            x: l(self.x, o.x),
            u: l(self.u, o.u),
            n: l(self.n, o.n).normalize(),
            roll: s(self.roll, o.roll),
            spin: s(self.spin, o.spin),
            loads: Loads {
                force: l(a.force, b.force),
                torque: l(a.torque, b.torque),
                on_water: l(a.on_water, b.on_water),
                spin_on_water: s(a.spin_on_water, b.spin_on_water),
                r_eff: s(a.r_eff, b.r_eff),
            },
        }
    }
}

/// Where the bullet's cavity opens through the water surface.
#[derive(Clone, Copy, Debug)]
pub struct Hole {
    pub center: Vec3,
    /// Outwards: against the path at the entry, along it at the exit.
    pub axis: Vec3,
    /// Source strength of the cavity.
    pub k: f64,
    /// When the leading end passed.
    pub t0: f64,
    pub r_eff: f64,
    /// Drag area in water, which grows many times over once the bullet turns side-on.
    pub drag_area: f64,
    pub speed: f64,
}

pub struct Path {
    /// Where the path enters the water, and the chord from there to where it leaves.
    pub origin: Vec3,
    pub dir: Vec3,
    pub a_out: f64,
    /// NPATH samples of 4 vec4f, see PathPoint in sim.wgsl.
    pub table: Vec<[f32; 4]>,
    pub entry: Hole,
    /// None when the bullet stops in the water.
    pub exit: Option<Hole>,
    /// When the bullet was through the water, or stopped in it.
    pub t_through: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    /// Gyroscopic stability at impact. Below 1 the bullet is tumbling by the time it
    /// arrives. Infinite for a ball.
    pub sg_air: f64,
    pub yaw_impact_deg: f64,
    pub max_yaw_water_deg: f64,
    pub v_impact: f64,
    pub v_exit: f64,
    pub stopped: bool,
    /// Whether the water phase has run yet. Later shots wait for the water they will meet.
    pub done: bool,
}

pub struct Flight {
    pub body: Body,
    /// Rifling turns per metre along the body.
    pub twist: f64,
    gravity: DVec3,
    /// ln(r_inf / r) of the cavity's source flow, as in the sim.
    log_r: f64,
    r_inf: f64,
    center: DVec3,
    pub poses: Vec<Pose>,
    /// Where the water phase takes over, and when.
    handoff: State,
    pub t_read: f64,
    /// When the tip reaches the balloon.
    pub t_hit: f64,
    pub dt_min: f64,
    pub path: Option<Path>,
    /// Chord coordinate of the leading end at each pose, never going back.
    front: Vec<f64>,
    pub report: Report,
}

impl Flight {
    /// Flies the bullet from the muzzle to `LOOKAHEAD` short of the balloon. It is aimed
    /// along `dir` at `target`, where its tip arrives at `t_hit` with the shot's speed: the
    /// aim allows for the drop and drift on the way, the muzzle speed for the air drag.
    pub fn new(s: &Shot, target: DVec3, dir: DVec3, t_hit: f64, gravity: f64, center: DVec3, r_inf: f64) -> Self {
        let body = Body::new(s);
        let log_r = (r_inf / body.radius).max(1.5).ln();
        let v_hit = (s.speed as f64).max(1.0);
        let mut f = Flight {
            body,
            twist: 0.0,
            gravity: DVec3::new(0.0, -gravity, 0.0),
            log_r,
            r_inf,
            center,
            poses: Vec::new(),
            handoff: State { x: target, u: dir, n: dir, h: DVec3::ZERO, roll: 0.0 },
            t_read: 0.0,
            t_hit,
            dt_min: (4e-6f64).min(1.2e-3 / v_hit),
            path: None,
            front: Vec::new(),
            report: Report::default(),
        };
        let p0 = s.spin_rpm as f64 / 60.0 * TAU;
        let yaw = (s.muzzle_yaw_deg as f64).to_radians();
        let range = (s.range_m as f64).max(0.5);
        let muzzle = target - dir * range;
        let launch = |f: &Flight, aim: DVec3, v: f64| {
            let side = (DVec3::Y - aim * aim.y).normalize_or(DVec3::Z);
            let n = aim * yaw.cos() + side * yaw.sin();
            State { x: muzzle - n * (f.body.length - f.body.u_cg), u: aim * v, n, h: n * (f.body.ix * p0), roll: 0.0 }
        };
        let crossing = |f: &Flight, st: &State| (f.tip(st) - target).dot(dir);
        let (mut aim, mut v_m) = (dir, v_hit);
        for _ in 0..8 {
            let (st, _) = f.fly_air(launch(&f, aim, v_m), 0.0, |st| crossing(&f, st), None);
            let miss = f.tip(&st) - target;
            let miss = miss - dir * miss.dot(dir);
            aim = (aim - miss / range).normalize();
            v_m *= v_hit / st.u.length();
            if miss.length() < 1e-4 && (st.u.length() - v_hit).abs() < 0.01 * v_hit {
                break;
            }
        }
        f.twist = p0 / (TAU * v_m);
        let mut poses = Vec::new();
        let (hand, t_hand) = f.fly_air(launch(&f, aim, v_m), 0.0, |st| crossing(&f, st) + LOOKAHEAD, Some(&mut poses));
        let (hit, t_c) = f.fly_air(hand, t_hand, |st| crossing(&f, st), None);
        if poses.is_empty() {
            poses.push(f.pose(t_hand, &hand, Loads::default()));
        }
        let shift = t_hit - t_c;
        for p in &mut poses {
            p.t += shift;
        }
        f.poses = poses;
        f.handoff = hand;
        f.t_read = t_hand + shift;
        let b = &f.body;
        let v = hit.u.length();
        let p = hit.h.dot(hit.n) / b.ix;
        let m_alpha = b.kappa_air * RHO_AIR * PI * b.radius * b.radius * v * v * (b.u_cp - b.u_cg);
        f.report.sg_air = if m_alpha > 0.0 { b.ix * b.ix * p * p / (4.0 * b.iy * m_alpha) } else { f64::INFINITY };
        f.report.yaw_impact_deg = hit.yaw().to_degrees();
        f.report.v_impact = v;
        f
    }

    fn tip(&self, s: &State) -> DVec3 {
        s.x + s.n * (self.body.length - self.body.u_cg)
    }

    fn lead(&self, x: DVec3, n: DVec3, u: DVec3) -> DVec3 {
        let b = &self.body;
        if n.dot(u) >= 0.0 {
            x + n * (b.length - b.u_cg)
        } else {
            x - n * b.u_cg
        }
    }

    fn loads(&self, s: &State, field: &Field) -> Loads {
        let b = &self.body;
        let n = s.n;
        let p = s.h.dot(n) / b.ix;
        let w_perp = (s.h - n * s.h.dot(n)) / b.iy;
        let at = |u: f64| s.x + n * (u - b.u_cg);
        let speed = s.u.length().max(1e-6);
        let uhat = s.u / speed;
        let cos_d = n.dot(uhat).clamp(-1.0, 1.0);
        let sin_d = (1.0 - cos_d * cos_d).sqrt();
        let nose_first = cos_d >= 0.0;
        let area = PI * b.radius * b.radius;
        let mut out = Loads { force: self.gravity * b.mass, ..Default::default() };
        // Cavitation number. Only once it is small does the nose sit in a vapour cavity with
        // the low drag that goes with it.
        let sigma = (2.0 * P_ATM / (RHO_WATER * speed * speed)).min(1.0);

        // The leading end, weighted by how much of it is in water.
        let lead = at(if nose_first { b.length } else { 0.0 });
        let (fill, v_water) = field.at(lead);
        let v_lead = s.u + w_perp.cross(lead - s.x);
        for water in [true, false] {
            let (rho, share, w) = if water { (RHO_WATER, fill, v_lead - v_water) } else { (RHO_AIR, 1.0 - fill, v_lead) };
            let wl = w.length();
            if share <= 0.0 || wl < 1e-9 {
                continue;
            }
            let wn = w.dot(n);
            let axial = |cd: f64| -0.5 * rho * cd * area * wn * wn.abs() * n;
            let none = (DVec3::ZERO, 0.0);
            // (force, where it acts)
            let parts = if !nose_first {
                [(axial(if water { CD_BASE_WATER * (1.0 + sigma) } else { CD_BASE_AIR }), 0.0), none]
            } else {
                let cd = if water { b.cd_water * (1.0 + sigma) } else { b.cd_air };
                match b.nose {
                    // A flat face is pushed square on, whatever the yaw.
                    Nose::Flat => [(axial(cd), b.length), none],
                    Nose::Sphere => [(-0.5 * rho * cd * area * wl * w, b.u_drag), none],
                    _ => {
                        let c = wn / wl;
                        let kappa = if water { b.kappa_water } else { b.kappa_air };
                        // In air all of the moment is in the calibrated lift.
                        let drag_u = if water { b.u_drag } else { b.u_cg };
                        [(-0.5 * rho * cd * area * c * c * wl * w, drag_u), (-rho * area * kappa * wn * (w - n * wn), b.u_cp)]
                    }
                }
            };
            for (f, u) in parts {
                out.force += f * share;
                out.torque += (at(u) - s.x).cross(f) * share;
                if water {
                    out.on_water -= f * share;
                }
            }
        }

        // The body behind the nose, in the cavity the nose opens. The cavity leaves the
        // shoulder, trails along the path and widens as a da/dt = K. K is from the nose's drag
        // as if it were in water, because behind a nose that has just come out the tail is
        // still in the cavity it made.
        let sep = at(if nose_first { b.u_shoulder } else { 0.0 });
        let cd_cav = if nose_first { b.cd_water } else { CD_BASE_WATER } * (1.0 + sigma);
        let k_cav = (0.25 * cd_cav * area * speed * speed / (PI * self.log_r)).sqrt();
        for &(u, du, r) in &b.slices {
            let xj = at(u);
            let vj = s.u + w_perp.cross(xj - s.x);
            let (fill, v_water) = field.at(xj);
            let mut wet = fill;
            let behind = (sep - xj).dot(uhat);
            if behind > 0.0 && wet > 0.0 {
                // How far the slice reaches out from the cavity's axis.
                let off = (xj - sep + uhat * behind).length();
                let reach = (off + r * cos_d.abs()).max(off.hypot(r));
                let rc = cavity(k_cav, behind / speed, b.radius).0;
                wet *= ((reach - rc) / r.max(1e-6)).clamp(0.0, 1.0);
            }
            for water in [true, false] {
                let (rho, share, w) = if water { (RHO_WATER, wet, vj - v_water) } else { (RHO_AIR, 1.0 - fill, vj) };
                if share <= 0.0 {
                    continue;
                }
                let wa = w.dot(n);
                let wc = w - n * wa;
                let wcl = wc.length();
                let mut f = DVec3::ZERO;
                // A ball's drag is all in its leading-end term.
                if b.nose != Nose::Sphere {
                    f -= 0.5 * rho * if water { CDC_WATER } else { CDC_AIR } * 2.0 * r * du * wcl * wc;
                }
                // Magnus: the spinning surface drags the cross flow round with it. The lift
                // grows with the spin ratio up to C_L = 1.
                if wcl > 1e-9 && p != 0.0 {
                    f += n.cross(wc) * (rho * r * r * p * du * (wcl / (r * p.abs())).min(1.0));
                }
                // In air skin friction is part of the drag coefficient.
                if water {
                    f -= n * (0.5 * rho * CF * TAU * r * du * wa.abs() * wa);
                }
                let spin = -0.5 * rho * CF * (p * r) * (p * r).abs() * TAU * r * du * r;
                out.force += f * share;
                out.torque += ((xj - s.x).cross(f) + n * spin) * share;
                if water {
                    out.on_water -= f * share;
                    out.spin_on_water -= spin * share;
                }
            }
        }
        let air = 1.0 - field.at(s.x).0;
        out.torque += w_perp * (0.5 * RHO_AIR * speed * area * 4.0 * b.radius * b.radius * CMQ_AIR * air);
        out.r_eff = if b.nose == Nose::Sphere { b.radius } else { ((area * cos_d.abs() + b.side_area * sin_d) / PI).sqrt() };
        out
    }

    fn deriv(&self, s: &State, field: &Field) -> (State, Loads) {
        let b = &self.body;
        let l = self.loads(s, field);
        let p = s.h.dot(s.n) / b.ix;
        let w_perp = (s.h - s.n * s.h.dot(s.n)) / b.iy;
        (State { x: s.u, u: l.force / b.mass, n: w_perp.cross(s.n), h: l.torque, roll: p }, l)
    }

    fn step(&self, s: &State, dt: f64, field: &Field) -> (State, Loads) {
        let (k1, l) = self.deriv(s, field);
        let (k2, _) = self.deriv(&s.add(&k1, 0.5 * dt), field);
        let (k3, _) = self.deriv(&s.add(&k2, 0.5 * dt), field);
        let (k4, _) = self.deriv(&s.add(&k3, dt), field);
        let mut o = s.add(&k1, dt / 6.0).add(&k2, dt / 3.0).add(&k3, dt / 3.0).add(&k4, dt / 6.0);
        o.n = o.n.normalize();
        (o, l)
    }

    /// A twentieth of a radian per step at the fastest rotation, from the tumble, the
    /// nutation or the fluid's pull on the body, and fine steps along the path in water.
    fn dt(&self, s: &State, fill: f64) -> f64 {
        let b = &self.body;
        let speed = s.u.length().max(1.0);
        let p = s.h.dot(s.n) / b.ix;
        let tumble = ((s.h - s.n * s.h.dot(s.n)) / b.iy).length();
        let rho = RHO_AIR + fill * RHO_WATER;
        let pull = (rho * PI * b.radius * b.radius * speed * speed * b.length / b.iy).sqrt();
        let ds = if fill > 0.0 { 2e-4 } else { 0.02 };
        (0.05 / (tumble + b.ix * p.abs() / b.iy + pull)).min(ds / speed)
    }

    /// Where the simulation takes over: centre of mass, velocity, axis, angular momentum and
    /// roll.
    pub fn handoff(&self) -> (DVec3, DVec3, DVec3, DVec3, f64) {
        let s = &self.handoff;
        (s.x, s.u, s.n, s.h, s.roll)
    }

    /// Adds where the simulation had the bullet at `t`, after every pose so far.
    pub fn push_pose(&mut self, t: f64, x: DVec3, u: DVec3, n: DVec3, roll: f64) {
        let spin = self.poses.last().map_or(0.0, |p| (roll - p.roll) / (t - p.t).max(1e-12));
        self.poses.push(Pose { t, x, u, n, roll, spin, loads: Loads::default() });
    }

    fn pose(&self, t: f64, s: &State, loads: Loads) -> Pose {
        Pose { t, x: s.x, u: s.u, n: s.n, roll: s.roll, spin: s.h.dot(s.n) / self.body.ix, loads }
    }

    /// Flies through still air until `g` turns positive, and returns the state where it
    /// crossed zero.
    fn fly_air(&self, mut s: State, mut t: f64, g: impl Fn(&State) -> f64, mut rec: Option<&mut Vec<Pose>>) -> (State, f64) {
        let air = Field::empty();
        let mut gv = g(&s);
        for _ in 0..2_000_000 {
            if gv >= 0.0 {
                break;
            }
            let dt = self.dt(&s, 0.0);
            let (next, loads) = self.step(&s, dt, &air);
            if let Some(r) = rec.as_deref_mut() {
                if (s.x - self.center).length() < NEAR {
                    r.push(self.pose(t, &s, loads));
                }
            }
            let gn = g(&next);
            if gn >= 0.0 {
                let k = -gv / (gn - gv);
                return (s.lerp(&next, k), t + dt * k);
            }
            s = next;
            t += dt;
            gv = gn;
        }
        (s, t)
    }

    /// Flies on from the handoff through `field`, the water as it is when the bullet gets
    /// there, and works out the path through it.
    pub fn through(&mut self, field: &Field) {
        let mut s = self.handoff;
        let mut t = self.t_read;
        let mut poses = Vec::new();
        let mut entry: Option<(DVec3, f64)> = None;
        let mut exit: Option<(DVec3, f64, f64)> = None;
        let mut stop: Option<f64> = None;
        let mut max_yaw = 0.0f64;
        let mut fill_prev = field.at(self.lead(s.x, s.n, s.u)).0;
        // Water that has spread out towards the gun can be there already.
        if fill_prev >= 0.5 {
            entry = Some((self.lead(s.x, s.n, s.u), t));
        }
        for _ in 0..5_000_000 {
            let lead = self.lead(s.x, s.n, s.u);
            let ahead = field.at(lead + s.u.normalize_or_zero() * 0.03).0;
            let dt = self.dt(&s, ahead.max(fill_prev).max(field.at(s.x).0));
            let (next, loads) = self.step(&s, dt, field);
            poses.push(self.pose(t, &s, loads));
            let lead_n = self.lead(next.x, next.n, next.u);
            let fill = field.at(lead_n).0;
            if (fill_prev < 0.5) != (fill < 0.5) {
                let k = (0.5 - fill_prev) / (fill - fill_prev);
                let at = lead + (lead_n - lead) * k;
                if fill >= 0.5 {
                    entry.get_or_insert((at, t + dt * k));
                    exit = None;
                } else {
                    exit = Some((at, t + dt * k, s.u.lerp(next.u, k).length()));
                }
            }
            s = next;
            t += dt;
            fill_prev = fill;
            let fill_cg = field.at(s.x).0;
            if fill_cg > 0.5 || fill >= 0.5 {
                max_yaw = max_yaw.max(s.yaw());
            }
            if s.u.length() < V_STOP && fill_cg > 0.5 {
                stop = Some(t);
                break;
            }
            if (s.x - self.center).length() > NEAR || s.x.y < -0.05 || t > T_END {
                break;
            }
        }
        poses.push(self.pose(t, &s, Loads::default()));
        if let Some(t_stop) = stop {
            // It stopped inside the water, which by now is falling freely with the burst
            // balloon, and sinks through it at its terminal speed until it reaches the ground.
            let b = &self.body;
            let area = if b.nose == Nose::Sphere { PI * b.radius * b.radius } else { 2.0 * b.radius * b.length };
            let v_sink = (2.0 * b.mass * -self.gravity.y / (RHO_WATER * area)).sqrt();
            let g = -self.gravity.y;
            let mut tau = 0.0;
            while t_stop + tau < T_END {
                tau += 1e-3;
                let y = s.x.y - 0.5 * g * tau * tau - v_sink * tau;
                let x = DVec3::new(s.x.x, y.max(b.radius), s.x.z);
                let u = if y > b.radius { DVec3::new(0.0, -g * tau - v_sink, 0.0) } else { DVec3::ZERO };
                poses.push(Pose { t: t_stop + tau, x, u, n: s.n, roll: s.roll, spin: 0.0, loads: Loads::default() });
            }
            exit = None;
        }
        self.poses.extend(poses);
        self.report.done = true;
        self.report.max_yaw_water_deg = max_yaw.to_degrees();
        self.report.stopped = stop.is_some();
        self.report.v_exit = exit.map_or(0.0, |e| e.2);
        let Some((p_in, _)) = entry else { return };
        let p_out = match (exit, stop) {
            (Some(e), _) => e.0,
            (None, Some(_)) => self.lead(s.x, s.n, s.u),
            // Still in water when it left the domain.
            (None, None) => return,
        };
        let t_through = exit.map_or(stop.unwrap_or(t), |e| e.1);
        self.build_path(p_in, p_out, exit.is_some(), t_through);
    }

    fn build_path(&mut self, p_in: DVec3, p_out: DVec3, exits: bool, t_through: f64) {
        let chord = p_out - p_in;
        let a_out = chord.length();
        if a_out < 1e-3 {
            return;
        }
        let dir = chord / a_out;
        let run_max = |v: Vec<f64>| {
            let mut m = f64::MIN;
            v.into_iter().map(|x| {
                m = m.max(x);
                m
            }).collect::<Vec<_>>()
        };
        let front = run_max(self.poses.iter().map(|p| (self.lead(p.x, p.n, p.u) - p_in).dot(dir)).collect());
        let cg = run_max(self.poses.iter().map(|p| (p.x - p_in).dot(dir)).collect());
        let at = |v: &[f64], a: f64| -> Pose {
            let i = v.partition_point(|&x| x < a);
            if i == 0 {
                return self.poses[0];
            }
            if i == v.len() {
                return *self.poses.last().unwrap();
            }
            let k = (a - v[i - 1]) / (v[i] - v[i - 1]).max(1e-12);
            self.poses[i - 1].lerp(&self.poses[i], k)
        };
        // The nose is only partly in water right at the surface, so the loads at the ends
        // are taken a grid half-cell inside.
        let edge = (2.5e-3f64).min(0.5 * a_out);
        let half_len = 0.5 * self.body.length;
        let mut table = Vec::with_capacity(NPATH * 4);
        let mut holes = Vec::new();
        for k in 0..NPATH {
            let a = a_out * k as f64 / (NPATH - 1) as f64;
            let tp = at(&front, a).t;
            // Where the path is from the centre of mass; what the water got from the bullet
            // when its leading end was there.
            let g = at(&cg, a);
            let l = at(&front, a.clamp(edge, a_out - edge));
            let off = g.x - (p_in + dir * a);
            let off = off - dir * off.dot(dir);
            let tan = g.u.normalize_or(dir);
            let r_eff = l.loads.r_eff.max(self.body.radius);
            let fw = l.loads.on_water;
            let k_src = (0.5 * fw.length() / (PI * RHO_WATER * (self.r_inf / r_eff).max(1.5).ln())).sqrt();
            let sig = 2.5 * r_eff;
            let along = l.u.dot(dir).max(0.2 * l.u.length()).max(1e-3);
            let mom = fw / (along * RHO_WATER * PI * sig * sig);
            let swirl = l.loads.spin_on_water * l.n.dot(tan) / along / (PI * RHO_WATER * sig.powi(4));
            let foot = (l.n - tan * l.n.dot(tan)) * half_len;
            table.push([off.x as f32, off.y as f32, off.z as f32, tp as f32]);
            table.push([tan.x as f32, tan.y as f32, tan.z as f32, k_src as f32]);
            table.push([mom.x as f32, mom.y as f32, mom.z as f32, swirl as f32]);
            table.push([foot.x as f32, foot.y as f32, foot.z as f32, r_eff as f32]);
            if k == 0 || k == NPATH - 1 {
                let speed = l.u.length().max(1e-3);
                let axis = if k == 0 { -tan } else { tan };
                holes.push(Hole {
                    center: (p_in + dir * a + off).as_vec3(),
                    axis: axis.as_vec3(),
                    k: k_src,
                    t0: tp,
                    r_eff,
                    drag_area: fw.length() / (0.5 * RHO_WATER * speed * speed),
                    speed,
                });
            }
        }
        self.front = front;
        self.path = Some(Path {
            origin: p_in.as_vec3(),
            dir: dir.as_vec3(),
            a_out,
            table,
            entry: holes[0],
            exit: exits.then_some(holes[1]),
            t_through,
        });
    }

    /// Chord coordinate of the leading end at `t`, never going back. Far behind the entry
    /// until the path is known.
    pub fn front(&self, t: f64) -> f64 {
        let Some(path) = &self.path else { return -1e9 };
        let i = self.poses.partition_point(|p| p.t <= t);
        if i == 0 {
            let p = &self.poses[0];
            return self.front[0] - p.u.length() * (p.t - t);
        }
        if i == self.poses.len() {
            let p = &self.poses[i - 1];
            return self.front[i - 1] + p.u.dot(path.dir.as_dvec3()).max(0.0) * (t - p.t);
        }
        let (a, b) = (&self.poses[i - 1], &self.poses[i]);
        let k = (t - a.t) / (b.t - a.t).max(1e-15);
        self.front[i - 1] + (self.front[i] - self.front[i - 1]) * k
    }

    /// Tip, unit axis towards it and roll angle at `t`.
    pub fn pose_at(&self, t: f64) -> (DVec3, DVec3, f64) {
        let b = &self.body;
        let i = self.poses.partition_point(|p| p.t <= t);
        let p = if i == 0 || i == self.poses.len() {
            let p = self.poses[i.min(self.poses.len() - 1)];
            Pose { x: p.x + p.u * (t - p.t), roll: p.roll + p.spin * (t - p.t), ..p }
        } else {
            let (a, c) = (&self.poses[i - 1], &self.poses[i]);
            a.lerp(c, (t - a.t) / (c.t - a.t).max(1e-15))
        };
        (p.x + p.n * (b.length - b.u_cg), p.n, p.roll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{SimParams, BULLETS};

    fn run(k: usize, tweak: impl Fn(&mut Shot)) -> Flight {
        let p = SimParams::default();
        let b = crate::balloon::build(&p);
        let mut s = Shot::default();
        s.set_bullet(BULLETS[k]);
        tweak(&mut s);
        let dir = DVec3::X;
        let target = b.entry(b.center, dir.as_vec3()).unwrap().as_dvec3();
        let mut f = Flight::new(&s, target, dir, 0.0, 9.81, b.center.as_dvec3(), b.radius_max);
        let parts: Vec<(Vec3, Vec3)> = b.particles.iter().map(|q| (Vec3::new(q[0], q[1], q[2]), Vec3::ZERO)).collect();
        f.through(&Field::new(&parts, b.spacing));
        let r = f.report;
        let path = f.path.as_ref().unwrap();
        println!(
            "{}: Sg {:.2}, yaw at impact {:.2}, max yaw in water {:.1}, v {:.0} -> {:.0}, stopped {}, chord {:.3} m, K in {:.3} out {:.3}, drag area x{:.1}",
            BULLETS[k].name,
            r.sg_air,
            r.yaw_impact_deg,
            r.max_yaw_water_deg,
            r.v_impact,
            r.v_exit,
            r.stopped,
            path.a_out,
            path.entry.k,
            path.exit.map_or(0.0, |h| h.k),
            path.exit.map_or(0.0, |h| h.drag_area) / path.entry.drag_area,
        );
        f
    }

    #[test]
    fn presets_fly_as_expected() {
        for k in 0..BULLETS.len() {
            let f = run(k, |_| {});
            assert!((f.report.v_impact - BULLETS[k].speed as f64).abs() < 2.0);
            assert!(f.report.sg_air > 1.0 || BULLETS[k].twist_in == 0.0);
        }
        // A long rifle bullet starts to tumble within a balloon's width, a pistol bullet only
        // yaws a little.
        assert!(run(3, |_| {}).report.max_yaw_water_deg > 30.0);
        assert!(run(1, |_| {}).report.max_yaw_water_deg < 10.0);
        // Without spin it is already tumbling when it arrives.
        assert!(run(3, |s| s.spin_rpm = 0.0).report.yaw_impact_deg > 10.0);
        // Slow and blunt, it stops inside.
        assert!(run(5, |s| {
            s.speed = 15.0;
            s.range_m = 1.0;
        })
        .report
        .stopped);
    }
}
