#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nose {
    Round,
    Spitzer,
    Flat,
    Sphere,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metal {
    Lead,
    Copper,
    Steel,
}

/// A real cartridge or airgun load. Picking one fills in caliber, mass, speed and spin,
/// which can then be changed one by one. Spin follows from the barrel twist: one turn per
/// twist length at the muzzle speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BulletType {
    pub name: &'static str,
    pub caliber_mm: f32,
    pub mass_g: f32,
    pub speed: f32,
    pub twist_in: f32,
    pub nose: Nose,
    pub metal: Metal,
}

impl BulletType {
    pub fn spin_rpm(&self) -> f32 {
        if self.twist_in <= 0.0 {
            return 0.0;
        }
        self.speed / (self.twist_in * 0.0254) * 60.0
    }
}

pub const LR22: BulletType = BulletType { name: ".22 LR, 40 gr lead round nose", caliber_mm: 5.7, mass_g: 2.59, speed: 330.0, twist_in: 16.0, nose: Nose::Round, metal: Metal::Lead };
pub const LUGER9: BulletType = BulletType { name: "9 mm Luger, 115 gr FMJ", caliber_mm: 9.0, mass_g: 7.45, speed: 360.0, twist_in: 10.0, nose: Nose::Round, metal: Metal::Copper };
pub const ACP45: BulletType = BulletType { name: ".45 ACP, 230 gr FMJ", caliber_mm: 11.5, mass_g: 14.9, speed: 255.0, twist_in: 16.0, nose: Nose::Round, metal: Metal::Copper };
pub const M855: BulletType = BulletType { name: "5.56 NATO, 62 gr M855", caliber_mm: 5.7, mass_g: 4.0, speed: 940.0, twist_in: 7.0, nose: Nose::Spitzer, metal: Metal::Copper };
pub const WIN308: BulletType = BulletType { name: ".308 Win, 150 gr spitzer", caliber_mm: 7.8, mass_g: 9.72, speed: 860.0, twist_in: 10.0, nose: Nose::Spitzer, metal: Metal::Copper };
pub const WADCUTTER: BulletType = BulletType { name: ".177 air rifle, 8.4 gr wadcutter", caliber_mm: 4.5, mass_g: 0.54, speed: 240.0, twist_in: 16.0, nose: Nose::Flat, metal: Metal::Lead };
pub const BB: BulletType = BulletType { name: ".177 steel BB", caliber_mm: 4.4, mass_g: 0.34, speed: 150.0, twist_in: 0.0, nose: Nose::Sphere, metal: Metal::Steel };

pub const BULLETS: [BulletType; 7] = [LR22, LUGER9, ACP45, M855, WIN308, WADCUTTER, BB];

pub const MAX_SHOTS: usize = 4;

/// One bullet fired at the balloon.
#[derive(Clone, Debug, PartialEq)]
pub struct Shot {
    pub bullet: BulletType,
    /// At impact. The bullet leaves the muzzle a little faster to make up for the air drag.
    pub speed: f32,
    pub caliber_mm: f32,
    pub mass_g: f32,
    /// At the muzzle.
    pub spin_rpm: f32,
    /// From the muzzle to the balloon. The yaw the bullet leaves the barrel with changes on
    /// the way.
    pub range_m: f32,
    /// Angle between the bullet's axis and its path as it leaves the muzzle.
    pub muzzle_yaw_deg: f32,
    /// Horizontal direction the shot comes from, around the balloon. 0 is from -x.
    pub azimuth_deg: f32,
    /// How far above horizontal the shot comes from. Negative shoots upwards.
    pub elevation_deg: f32,
    pub impact_offset_cm: f32,
    pub impact_side_cm: f32,
    /// When the bullet reaches the balloon, counted from the earliest shot.
    pub delay_ms: f32,
}

impl Shot {
    pub fn with_bullet(b: BulletType) -> Self {
        let mut s = Self::default();
        s.set_bullet(b);
        s
    }

    pub fn set_bullet(&mut self, b: BulletType) {
        self.bullet = b;
        self.caliber_mm = b.caliber_mm;
        self.mass_g = b.mass_g;
        self.speed = b.speed;
        self.spin_rpm = b.spin_rpm();
    }
}

impl Default for Shot {
    fn default() -> Self {
        let b = LR22;
        Self {
            bullet: b,
            speed: b.speed,
            caliber_mm: b.caliber_mm,
            mass_g: b.mass_g,
            spin_rpm: b.spin_rpm(),
            range_m: 10.0,
            muzzle_yaw_deg: 0.5,
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
            impact_offset_cm: 0.0,
            impact_side_cm: 0.0,
            delay_ms: 0.0,
        }
    }
}

/// Parameters that change the simulation. Editing any of these requires a re-simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct SimParams {
    /// At least one, at most MAX_SHOTS.
    pub shots: Vec<Shot>,
    pub balloon_diameter_cm: f32,
    pub balloon_height: f32,
    pub rubber_tension: f32,
    pub rubber_stretch: f32,
    pub rubber_strength: f32,
    pub rubber_thickness_mm: f32,
    pub gravity: f32,
    pub spray_amount: f32,
    pub mist_amount: f32,
    pub resolution: u32,
}

impl SimParams {
    /// A copy of the last shot, coming from a quarter turn further round so it makes its own
    /// holes.
    pub fn add_shot(&mut self) {
        let mut s = self.shots.last().cloned().unwrap_or_default();
        s.azimuth_deg = (s.azimuth_deg + 270.0) % 360.0 - 180.0;
        self.shots.push(s);
    }
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            shots: vec![Shot::default()],
            balloon_diameter_cm: 16.0,
            balloon_height: 0.55,
            rubber_tension: 140.0,
            rubber_stretch: 2.2,
            rubber_strength: 1.5,
            rubber_thickness_mm: 0.25,
            gravity: 9.81,
            spray_amount: 1.0,
            mist_amount: 1.0,
            resolution: 64,
        }
    }
}

/// Parameters that only change how the scene is rendered.
#[derive(Clone, Debug, PartialEq)]
pub struct LookParams {
    pub balloon_color: [f32; 3],
    pub sun_intensity: f32,
    pub sky_intensity: f32,
    pub env_rotation_deg: f32,
    pub exposure_ev: f32,
    pub dof: bool,
    pub f_stop: f32,
    pub focal_mm: f32,
    pub bloom: f32,
    pub caustics: bool,
    pub mist: bool,
    pub render_scale: f32,
    pub max_bounces: u32,
    /// Exposure time of the camera, which sets the motion blur of fast drops.
    pub shutter_us: f32,
}

impl Default for LookParams {
    fn default() -> Self {
        Self {
            balloon_color: [0.85, 0.06, 0.05],
            sun_intensity: 1.0,
            sky_intensity: 1.0,
            env_rotation_deg: 0.0,
            exposure_ev: 0.0,
            dof: true,
            f_stop: 4.0,
            focal_mm: 50.0,
            bloom: 0.04,
            caustics: true,
            mist: true,
            render_scale: 1.0,
            max_bounces: 8,
            shutter_us: 20.0,
        }
    }
}

fn parse<T: std::str::FromStr>(v: &str) -> anyhow::Result<T> {
    v.parse::<T>().map_err(|_| anyhow::anyhow!("bad value '{v}'"))
}

/// Applies a `key=value` override from the command line. Bullet keys go to the first shot,
/// or to shot N with a `shotN.` prefix. `shots=N` sets how many there are.
pub fn set_kv(sim: &mut SimParams, look: &mut LookParams, key: &str, v: &str) -> anyhow::Result<()> {
    let (idx, name) = match key.strip_prefix("shot").and_then(|r| r.split_once('.')) {
        Some((n, name)) => (parse::<usize>(n)?.checked_sub(1).ok_or_else(|| anyhow::anyhow!("shots count from 1"))?, name),
        None => (0, key),
    };
    anyhow::ensure!(idx < MAX_SHOTS, "at most {MAX_SHOTS} shots");
    let count = if name == "shots" { parse::<usize>(v)? } else { sim.shots.len().max(idx + 1) };
    anyhow::ensure!((1..=MAX_SHOTS).contains(&count), "shots 1..{MAX_SHOTS}");
    sim.shots.truncate(count);
    while sim.shots.len() < count {
        sim.add_shot();
    }
    let s = &mut sim.shots[idx];
    match name {
        "shots" => {}
        "bullet" => {
            let k: usize = parse(v)?;
            s.set_bullet(*BULLETS.get(k).ok_or_else(|| anyhow::anyhow!("bullet 0..{}", BULLETS.len() - 1))?);
        }
        "bullet_speed" => s.speed = parse(v)?,
        "caliber" => s.caliber_mm = parse(v)?,
        "mass" => s.mass_g = parse(v)?,
        "spin" => s.spin_rpm = parse(v)?,
        "range" => s.range_m = parse(v)?,
        "yaw" => s.muzzle_yaw_deg = parse(v)?,
        "azimuth" => s.azimuth_deg = parse(v)?,
        "elevation" => s.elevation_deg = parse(v)?,
        "impact_offset" => s.impact_offset_cm = parse(v)?,
        "impact_side" => s.impact_side_cm = parse(v)?,
        "delay" => s.delay_ms = parse(v)?,
        "diameter" => sim.balloon_diameter_cm = parse(v)?,
        "height" => sim.balloon_height = parse(v)?,
        "tension" => sim.rubber_tension = parse(v)?,
        "stretch" => sim.rubber_stretch = parse(v)?,
        "strength" => sim.rubber_strength = parse(v)?,
        "thickness" => sim.rubber_thickness_mm = parse(v)?,
        "gravity" => sim.gravity = parse(v)?,
        "spray" => sim.spray_amount = parse(v)?,
        "mist" => sim.mist_amount = parse(v)?,
        "resolution" => sim.resolution = parse(v)?,
        "sun" => look.sun_intensity = parse(v)?,
        "sky" => look.sky_intensity = parse(v)?,
        "env_rotation" => look.env_rotation_deg = parse(v)?,
        "exposure" => look.exposure_ev = parse(v)?,
        "dof" => look.dof = parse(v)?,
        "fstop" => look.f_stop = parse(v)?,
        "focal" => look.focal_mm = parse(v)?,
        "bloom" => look.bloom = parse(v)?,
        "caustics" => look.caustics = parse(v)?,
        "mist_render" => look.mist = parse(v)?,
        "shutter" => look.shutter_us = parse(v)?,
        "max_bounces" => look.max_bounces = parse(v)?,
        "color" => {
            let c: Vec<f32> = v.split(',').map(parse).collect::<anyhow::Result<_>>()?;
            anyhow::ensure!(c.len() == 3, "color needs r,g,b");
            look.balloon_color = [c[0], c[1], c[2]];
        }
        _ => anyhow::bail!("unknown parameter '{key}'"),
    }
    Ok(())
}
