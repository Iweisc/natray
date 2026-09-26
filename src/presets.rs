//! Ready-made scenes: what to simulate, how it looks and where the camera starts.

use crate::params::{LookParams, Shot, SimParams, M855, WADCUTTER};

#[derive(Clone, Copy)]
pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

pub struct Preset {
    pub name: &'static str,
    pub about: &'static str,
    pub sim: SimParams,
    pub look: LookParams,
    pub view: View,
}

pub fn all() -> Vec<Preset> {
    vec![
        Preset {
            name: "Backlit",
            about: "The default shot with the sun behind the balloon, so the spray and mist light up.",
            sim: SimParams::default(),
            look: LookParams { env_rotation_deg: 95.0, bloom: 0.08, exposure_ev: 0.3, ..LookParams::default() },
            view: View { yaw: -35.0, pitch: 6.0, dist: 0.9 },
        },
        Preset {
            name: "Tumbling rifle round",
            about: "A long rifle bullet turns sideways in the water, so the exit blows out far wider than the entry.",
            sim: SimParams {
                shots: vec![Shot { speed: 500.0, ..Shot::with_bullet(M855) }],
                balloon_diameter_cm: 22.0,
                ..SimParams::default()
            },
            look: LookParams { balloon_color: [0.05, 0.25, 0.9], ..LookParams::default() },
            view: View { yaw: -20.0, pitch: 5.0, dist: 2.0 },
        },
        Preset {
            name: "Crossfire",
            about: "Two bullets at right angles, at the same moment, one just above the other.",
            sim: SimParams {
                shots: vec![
                    Shot { impact_offset_cm: 1.5, ..Shot::default() },
                    Shot { azimuth_deg: 90.0, impact_offset_cm: -1.5, ..Shot::default() },
                ],
                ..SimParams::default()
            },
            look: LookParams { balloon_color: [0.95, 0.45, 0.05], ..LookParams::default() },
            view: View { yaw: -45.0, pitch: 12.0, dist: 1.3 },
        },
        Preset {
            name: "Four sides",
            about: "Four bullets half a millisecond apart, each from a quarter turn further round.",
            sim: SimParams {
                shots: [(0.0, 0.0), (90.0, 0.5), (-180.0, 1.0), (-90.0, 1.5)]
                    .into_iter()
                    .map(|(azimuth_deg, delay_ms)| Shot { azimuth_deg, delay_ms, ..Shot::default() })
                    .collect(),
                ..SimParams::default()
            },
            look: LookParams { balloon_color: [0.1, 0.7, 0.2], ..LookParams::default() },
            view: View { yaw: -35.0, pitch: 20.0, dist: 1.2 },
        },
        Preset {
            name: "Moon gravity",
            about: "One sixth of Earth's gravity: the water hangs in the air and comes apart slowly.",
            sim: SimParams { gravity: 1.62, balloon_diameter_cm: 20.0, ..SimParams::default() },
            look: LookParams { balloon_color: [0.95, 0.8, 0.05], ..LookParams::default() },
            view: View { yaw: -35.0, pitch: 4.0, dist: 1.1 },
        },
        Preset {
            name: "Weightless",
            about: "No gravity. The balloon is a sphere and the water flies apart without falling.",
            sim: SimParams { gravity: 0.0, ..SimParams::default() },
            look: LookParams { balloon_color: [0.55, 0.1, 0.8], ..LookParams::default() },
            view: View { yaw: -35.0, pitch: 8.0, dist: 1.0 },
        },
        Preset {
            name: "Slow pellet",
            about: "A slow airgun pellet. Weak, tight rubber tears all the way round and the water keeps the balloon's shape for a moment.",
            sim: SimParams {
                shots: vec![Shot { speed: 120.0, ..Shot::with_bullet(WADCUTTER) }],
                rubber_strength: 1.15,
                rubber_tension: 200.0,
                ..SimParams::default()
            },
            look: LookParams { balloon_color: [0.9, 0.1, 0.45], ..LookParams::default() },
            view: View { yaw: -35.0, pitch: 8.0, dist: 0.8 },
        },
    ]
}

pub fn named(name: &str) -> anyhow::Result<Preset> {
    let all = all();
    let names = all.iter().map(|p| p.name).collect::<Vec<_>>().join(", ");
    all.into_iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| anyhow::anyhow!("no preset '{name}', try one of: {names}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::MAX_SHOTS;

    #[test]
    fn presets_fit_the_shot_limit() {
        for p in all() {
            assert!((1..=MAX_SHOTS).contains(&p.sim.shots.len()), "{}", p.name);
        }
    }
}
