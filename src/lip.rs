//! The cavity lip at the entry and exit holes: how fast the cavity opens there and when it
//! throws off spray. shaders/lip.wgsl has the same formulas for the emitter and the crown.

use glam::Vec3;

pub const P_ATM: f64 = 101325.0;
/// Air vents into the cavity through the holes over about this time. Longer and the cavity
/// collapses sooner and harder. Tuned by eye against the usual open-then-collapse sequence.
pub const TAU_VENT: f64 = 6e-3;
/// Lip material slower than this stays with the water body instead of leaving as spray.
const V_MIN: f64 = 1.5;

pub fn cavity(k: f64, tau: f64, rb: f64) -> (f64, f64) {
    let b = P_ATM * TAU_VENT / 1500.0;
    let e = (-tau / TAU_VENT).exp();
    let a = (rb * rb + 2.0 * (k * tau - b * (tau - TAU_VENT * (1.0 - e)))).max(0.0).sqrt();
    (a, (k - b * (1.0 - e)) / a.max(1e-6))
}

fn lip_speed(wall: f64) -> f64 {
    wall * 1.15f64.hypot(2.2)
}

#[derive(Clone, Copy, Debug)]
pub struct Lip {
    pub center: Vec3,
    /// Outwards: against the bullet's path at the entry, along it at the exit.
    pub axis: Vec3,
    /// Source strength of the cavity at the hole.
    pub k: f64,
    /// When the bullet passed the hole.
    pub t0: f64,
    /// How long after that the lip throws off spray.
    pub tau_end: f64,
    pub rb: f64,
}

impl Lip {
    pub fn new(center: Vec3, axis: Vec3, k: f64, t0: f64, rb: f64) -> Self {
        // The wall only slows down, so the cut-off is a bisection. A cavity that is still
        // fast after 20 ms belongs to a rifle round that has blown the balloon apart anyway.
        let fast = |tau| lip_speed(cavity(k, tau, rb).1) >= V_MIN;
        let (mut lo, mut hi) = (0.0, 0.02);
        if fast(hi) {
            lo = hi;
        }
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if fast(m) {
                lo = m;
            } else {
                hi = m;
            }
        }
        Self { center, axis, k, t0, tau_end: lo, rb }
    }

    fn swept(&self, tau: f64) -> f64 {
        let a = cavity(self.k, tau, self.rb).0;
        a * a - self.rb * self.rb
    }

    /// Share of this lip's spray thrown off by time `t`. It follows the area the cavity
    /// wall has swept, since that is the water pushed out past the lip.
    pub fn emitted(&self, t: f64) -> f64 {
        let tau = (t - self.t0).clamp(0.0, self.tau_end);
        self.swept(tau) / self.swept(self.tau_end).max(1e-12)
    }
}
