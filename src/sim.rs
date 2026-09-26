use crate::balloon::{self, Balloon, RHO_RUBBER, RHO_WATER};
use crate::flight::{Field, Flight, NPATH};
use crate::gpu::{bgl, bind, groups_1d, Gpu, B};
use crate::lip::{Lip, P_ATM, TAU_VENT};
use crate::params::{SimParams, MAX_SHOTS};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

const STEP_STRIDE: u64 = 256;
const MAX_BATCH: usize = 512;
pub const SPRAY_CAP: u32 = 1_200_000;
pub const WET_RES: u32 = 1024;
pub const WET_HALF: f32 = 1.2;
/// Artificial sound speed of the weakly compressible water.
const SOUND: f64 = 20.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Step {
    t: f32,
    dt: f32,
    dt_m: f32,
    n_particles: u32,
    gravity: [f32; 4],
    grid_origin: [f32; 4],
    grid_dims: [u32; 4],
    /// Each shot's front at the start and the end of the step.
    front: [[f32; 4]; 2],
    /// Spray thrown off at each hole: entry and exit of shot 0, then of shot 1, ...
    emit: [[u32; 4]; 2],
    r_inf: f32,
    p_cav: f32,
    tau_vent: f32,
    v_cap: f32,
    spacing: f32,
    n_tris: u32,
    n_emit: u32,
    spray_cap: u32,
    n_shots: u32,
    _pad: [u32; 3],
    mem: [f32; 4],
    spray: [f32; 4],
    seed: [u32; 4],
    misc: [f32; 4],
}

/// A shot as the sim shaders see it, see Shot in sim.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShotG {
    origin: [f32; 4],
    dir: [f32; 4],
    mouth: [[f32; 4]; 4],
    info: [f32; 4],
}

pub struct ShotRun {
    pub flight: Flight,
    /// Entry and exit hole, once the water on the path is known.
    pub lips: [Option<Lip>; 2],
    emit_total: [f64; 2],
    emitted: [u64; 2],
}

struct Pipes {
    region_prep: wgpu::ComputePipeline,
    grid_clear: wgpu::ComputePipeline,
    p2g: wgpu::ComputePipeline,
    grid_update: wgpu::ComputePipeline,
    g2p: wgpu::ComputePipeline,
    mem_step: wgpu::ComputePipeline,
    mem_break: wgpu::ComputePipeline,
    spray_emit: wgpu::ComputePipeline,
    spray_update: wgpu::ComputePipeline,
}

pub struct Sim {
    pub params: SimParams,
    pub scene: Balloon,
    pub n_particles: u32,
    pub n_verts: u32,
    pub n_tris: u32,
    pub dx: f64,
    pub grid_origin: Vec3,
    pub dims: [u32; 3],

    pub particles: wgpu::Buffer,
    /// Holds the spray slot count at word 16.
    pub state: wgpu::Buffer,
    indirect: wgpu::Buffer,
    pub mem_pos: wgpu::Buffer,
    pub tris: wgpu::Buffer,
    pub alive: wgpu::Buffer,
    pub vtri_off: wgpu::Buffer,
    pub vtri: wgpu::Buffer,
    pub spray: wgpu::Buffer,
    pub touch: wgpu::Buffer,
    pub landed: wgpu::Buffer,
    shots_buf: wgpu::Buffer,
    path_buf: wgpu::Buffer,
    step_buf: wgpu::Buffer,
    bg: [wgpu::BindGroup; 2],
    ind_bg: wgpu::BindGroup,
    pipes: Pipes,

    pub t: f64,
    pub t_start: f64,
    dt: f64,
    pub step_idx: u64,
    started: bool,
    dt_max: f64,
    /// When the first bullet was through the balloon and it burst.
    t_pop: f64,
    pub shots: Vec<ShotRun>,
}

impl Sim {
    pub fn new(gpu: &Gpu, params: &SimParams) -> Self {
        let scene = balloon::build(params);
        let n_particles = scene.particles.len() as u32;
        let n_verts = scene.mem_pos.len() as u32;
        let n_tris = scene.tris.len() as u32;
        let dx = 2.0 * scene.spacing;

        let top = scene.bottom_y + scene.profile.z_top() + 0.08;
        let ny = ((top / dx).ceil() as u32 + 4).max(16);
        let mut nxz = ((1.2 / dx).ceil() as u32).max(32);
        while (nxz as u64) * (nxz as u64) * ny as u64 > 16_000_000 {
            nxz -= 2;
        }
        let dims = [nxz, ny, nxz];
        let grid_origin = Vec3::new(-(nxz as f32) * 0.5 * dx as f32, -3.0 * dx as f32, -(nxz as f32) * 0.5 * dx as f32);

        let particles_init: Vec<[f32; 20]> = scene
            .particles
            .iter()
            .map(|p| {
                let mut a = [0f32; 20];
                a[..4].copy_from_slice(p);
                a[7] = 1.0;
                a
            })
            .collect();
        let particles = gpu.storage_init("particles", bytemuck::cast_slice(&particles_init));
        let n_nodes = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
        let grid = gpu.storage("mpm grid", n_nodes * 16);
        let state = gpu.storage("sim state", 80);
        let indirect = gpu.buffer(
            "indirect",
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        );
        let mem_pos = gpu.storage_init("mem pos a", bytemuck::cast_slice(&scene.mem_pos));
        let mem_pos_b = gpu.storage_init("mem pos b", bytemuck::cast_slice(&scene.mem_pos));
        let zeros = vec![[0f32; 4]; n_verts as usize];
        let mem_vel = [
            gpu.storage_init("mem vel a", bytemuck::cast_slice(&zeros)),
            gpu.storage_init("mem vel b", bytemuck::cast_slice(&zeros)),
        ];
        let vinfo = gpu.storage_init("vinfo", bytemuck::cast_slice(&scene.vinfo));
        let adj = gpu.storage_init("adj", bytemuck::cast_slice(&scene.adj));
        let edges = gpu.storage_init("edges", bytemuck::cast_slice(&scene.edges));
        let tris = gpu.storage_init("tris", bytemuck::cast_slice(&scene.tris));
        let alive = gpu.storage_init("alive", bytemuck::cast_slice(&vec![1u32; n_tris as usize]));
        let vtri_off = gpu.storage_init("vtri off", bytemuck::cast_slice(&scene.vtri_off));
        let vtri = gpu.storage_init("vtri", bytemuck::cast_slice(&scene.vtri));
        let spray = gpu.storage("spray", SPRAY_CAP as u64 * 48);
        let touch = gpu.storage_init("ground touch", bytemuck::cast_slice(&vec![u32::MAX; (WET_RES * WET_RES) as usize]));
        let landed = gpu.storage("landed", SPRAY_CAP as u64 * 16);
        let shots_buf = gpu.storage("shots", (MAX_SHOTS * std::mem::size_of::<ShotG>()) as u64);
        let path_buf = gpu.storage("shot paths", (MAX_SHOTS * NPATH * 64) as u64);
        let hit_r = gpu.storage("hit radius", n_particles as u64 * 16);
        let step_buf = gpu.buffer(
            "steps",
            STEP_STRIDE * MAX_BATCH as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        use B::*;
        let layout = bgl(
            gpu,
            "sim",
            &[UniformDyn, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Ro, Ro, Ro, Ro, Rw, Rw, Rw, Rw, Ro, Ro, Rw],
            wgpu::ShaderStages::COMPUTE,
        );
        let step_res = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &step_buf,
            offset: 0,
            size: wgpu::BufferSize::new(STEP_STRIDE),
        });
        let mk = |pa: &wgpu::Buffer, va: &wgpu::Buffer, pb: &wgpu::Buffer, vb: &wgpu::Buffer| {
            bind(
                gpu,
                &layout,
                &[
                    step_res.clone(),
                    particles.as_entire_binding(),
                    grid.as_entire_binding(),
                    state.as_entire_binding(),
                    pa.as_entire_binding(),
                    va.as_entire_binding(),
                    pb.as_entire_binding(),
                    vb.as_entire_binding(),
                    vinfo.as_entire_binding(),
                    adj.as_entire_binding(),
                    edges.as_entire_binding(),
                    tris.as_entire_binding(),
                    alive.as_entire_binding(),
                    spray.as_entire_binding(),
                    touch.as_entire_binding(),
                    landed.as_entire_binding(),
                    shots_buf.as_entire_binding(),
                    path_buf.as_entire_binding(),
                    hit_r.as_entire_binding(),
                ],
            )
        };
        let bg = [mk(&mem_pos, &mem_vel[0], &mem_pos_b, &mem_vel[1]), mk(&mem_pos_b, &mem_vel[1], &mem_pos, &mem_vel[0])];
        let module = gpu.shader("sim", &[include_str!("../shaders/common.wgsl"), include_str!("../shaders/lip.wgsl"), include_str!("../shaders/sim.wgsl")]);
        let ind_layout = bgl(gpu, "sim indirect", &[Rw], wgpu::ShaderStages::COMPUTE);
        let ind_bg = bind(gpu, &ind_layout, &[indirect.as_entire_binding()]);
        let pl = gpu.layout(&[&layout]);
        let pipes = Pipes {
            // Only this pipeline sees the indirect-args buffer as storage; the others use it
            // for indirect dispatch, which may not overlap a storage binding.
            region_prep: gpu.compute(&gpu.layout(&[&layout, &ind_layout]), &module, "region_prep"),
            grid_clear: gpu.compute(&pl, &module, "grid_clear"),
            p2g: gpu.compute(&pl, &module, "p2g"),
            grid_update: gpu.compute(&pl, &module, "grid_update"),
            g2p: gpu.compute(&pl, &module, "g2p"),
            mem_step: gpu.compute(&pl, &module, "mem_step"),
            mem_break: gpu.compute(&pl, &module, "mem_break"),
            spray_emit: gpu.compute(&pl, &module, "spray_emit"),
            spray_update: gpu.compute(&pl, &module, "spray_update"),
        };

        let dt_max = 0.5 * dx / (SOUND + 22.0);
        let first = params.shots.iter().map(|s| s.delay_ms).fold(f32::MAX, f32::min);
        let mut shots: Vec<ShotRun> = params
            .shots
            .iter()
            .map(|s| {
                let (az, el) = (s.azimuth_deg.to_radians(), s.elevation_deg.to_radians());
                let dir = Vec3::new(az.cos() * el.cos(), -el.sin(), az.sin() * el.cos());
                let side = dir.cross(Vec3::Y).normalize_or(Vec3::Z);
                let aim = scene.center + Vec3::Y * (s.impact_offset_cm * 0.01) + side * (s.impact_side_cm * 0.01);
                // A line that misses the balloon is timed as if it met it a radius short of the aim.
                let target = scene.entry(aim, dir).unwrap_or(aim - dir * scene.radius_max as f32);
                let t_hit = (s.delay_ms - first) as f64 * 1e-3;
                let flight = Flight::new(s, target.as_dvec3(), dir.as_dvec3(), t_hit, params.gravity as f64, scene.center.as_dvec3(), scene.radius_max);
                ShotRun { flight, lips: [None, None], emit_total: [0.0; 2], emitted: [0; 2] }
            })
            .collect();
        let t_start = shots.iter().map(|s| s.flight.t_read - (0.3 - crate::flight::LOOKAHEAD) / s.flight.report.v_impact).fold(0.0, f64::min);
        // Bullets that get there before anything has moved see the balloon as built.
        let still: Vec<(Vec3, Vec3)> = scene.particles.iter().map(|p| (Vec3::new(p[0], p[1], p[2]), Vec3::ZERO)).collect();
        let field = Field::new(&still, scene.spacing);
        for s in shots.iter_mut().filter(|s| s.flight.t_read <= 0.0) {
            s.flight.through(&field);
        }

        let mut sim = Self {
            params: params.clone(),
            n_particles,
            n_verts,
            n_tris,
            dx,
            grid_origin,
            dims,
            particles,
            state,
            indirect,
            mem_pos,
            tris,
            alive,
            vtri_off,
            vtri,
            spray,
            touch,
            landed,
            shots_buf,
            path_buf,
            step_buf,
            bg,
            ind_bg,
            pipes,
            t: t_start,
            t_start,
            dt: dt_max,
            step_idx: 0,
            started: false,
            dt_max,
            t_pop: f64::INFINITY,
            shots,
            scene,
        };
        sim.write_state(gpu);
        for i in 0..sim.shots.len() {
            sim.open_holes(gpu, i);
        }
        sim
    }

    /// Sets up the holes and spray of a shot whose path through the water is known, and
    /// hands it to the shaders.
    fn open_holes(&mut self, gpu: &Gpu, i: usize) {
        let spray = self.params.spray_amount as f64;
        let s = &mut self.shots[i];
        let mut g = ShotG::zeroed();
        if let Some(p) = &s.flight.path {
            let e = p.entry;
            let e_fac = (0.5 * RHO_WATER * e.drag_area * e.speed * e.speed / 1306.0).clamp(0.2, 4.0);
            s.lips[0] = Some(Lip::new(e.center, e.axis, e.k, e.t0, e.r_eff));
            s.emit_total[0] = 30000.0 * spray * e_fac;
            g.origin = p.origin.extend(s.flight.body.radius as f32).into();
            g.dir = p.dir.extend(p.a_out as f32).into();
            g.mouth[0] = e.center.extend(e.k as f32).into();
            g.mouth[1] = e.axis.extend(e.t0 as f32).into();
            g.info = [0.0, e.r_eff as f32, 0.0, (1.25 * s.flight.body.radius) as f32];
            if let Some(x) = p.exit {
                s.lips[1] = Some(Lip::new(x.center, x.axis, x.k, x.t0, x.r_eff));
                // A bullet that comes out side-on throws far more water than one still nose first.
                s.emit_total[1] = 80000.0 * spray * (e_fac * x.drag_area / e.drag_area).min(8.0);
                g.mouth[2] = x.center.extend(x.k as f32).into();
                g.mouth[3] = x.axis.extend(x.t0 as f32).into();
                g.info[0] = x.speed as f32;
                g.info[2] = x.r_eff as f32;
            }
            self.t_pop = self.t_pop.min(p.t_through + 4e-4);
            gpu.queue.write_buffer(&self.path_buf, (i * NPATH * 64) as u64, bytemuck::cast_slice(&p.table));
        }
        gpu.queue.write_buffer(&self.shots_buf, (i * std::mem::size_of::<ShotG>()) as u64, bytemuck::bytes_of(&g));
    }

    /// Reads back the water as it is now, for a bullet about to reach it.
    fn enter_water(&mut self, gpu: &Gpu, i: usize) {
        let raw = crate::gpu::read_buffer(gpu, &self.particles, self.n_particles as u64 * 80);
        let parts: Vec<(Vec3, Vec3)> = bytemuck::cast_slice::<u8, [f32; 20]>(&raw)
            .iter()
            .filter(|p| p[7] > 0.5)
            .map(|p| (Vec3::new(p[0], p[1], p[2]), Vec3::new(p[4], p[5], p[6])))
            .collect();
        self.shots[i].flight.through(&Field::new(&parts, self.scene.spacing));
        self.open_holes(gpu, i);
    }

    /// The next shot still waiting for the water it will meet, and when it needs it.
    fn next_read(&self) -> Option<(usize, f64)> {
        self.shots.iter().enumerate().filter(|(_, s)| !s.flight.report.done).map(|(i, s)| (i, s.flight.t_read)).min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The small step while a bullet is close to or inside the water, and a while after.
    fn fast_dt(&self, t: f64) -> Option<f64> {
        self.shots
            .iter()
            .filter_map(|s| {
                let p = s.flight.path.as_ref()?;
                (t >= s.flight.t_read && t < p.t_through + 4e-4).then_some(s.flight.dt_min)
            })
            .reduce(f64::min)
    }

    fn write_state(&self, gpu: &Gpu) {
        let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for p in &self.scene.particles {
            for k in 0..3 {
                let c = ((p[k] - self.grid_origin[k]) as f64 / self.dx - 0.5).floor() as i32;
                lo[k] = lo[k].min(c);
                hi[k] = hi[k].max(c);
            }
        }
        let mut s = [0i32; 20];
        s[..3].copy_from_slice(&lo);
        s[4..7].copy_from_slice(&hi);
        gpu.queue.write_buffer(&self.state, 0, bytemuck::cast_slice(&s));
    }

    pub fn debug_stats(&self, gpu: &Gpu) {
        let alive: Vec<u32> = bytemuck::cast_slice(&crate::gpu::read_buffer(gpu, &self.alive, self.n_tris as u64 * 4)).to_vec();
        let pos: Vec<[f32; 4]> = bytemuck::cast_slice(&crate::gpu::read_buffer(gpu, &self.mem_pos, self.n_verts as u64 * 16)).to_vec();
        let nan = pos.iter().filter(|p| !p[0].is_finite()).count();
        let (mut lo, mut hi) = (Vec3::splat(1e9), Vec3::splat(-1e9));
        let mut disp = 0.0f32;
        for (p, p0) in pos.iter().zip(&self.scene.mem_pos) {
            let v = Vec3::new(p[0], p[1], p[2]);
            lo = lo.min(v);
            hi = hi.max(v);
            disp = disp.max((v - Vec3::new(p0[0], p0[1], p0[2])).length());
        }
        let parts: Vec<[f32; 20]> = bytemuck::cast_slice(&crate::gpu::read_buffer(gpu, &self.particles, self.n_particles as u64 * 80)).to_vec();
        let live = parts.iter().filter(|p| p[7] > 0.5).count();
        let vmax = parts.iter().filter(|p| p[7] > 0.5).map(|p| Vec3::new(p[4], p[5], p[6]).length()).fold(0.0, f32::max);
        let (mut ylo, mut yhi, mut ysum, mut vysum) = (f32::MAX, f32::MIN, 0.0, 0.0);
        for p in parts.iter().filter(|p| p[7] > 0.5) {
            ylo = ylo.min(p[1]);
            yhi = yhi.max(p[1]);
            ysum += p[1];
            vysum += p[5];
        }
        let n = live.max(1) as f32;
        let st: Vec<u32> = bytemuck::cast_slice(&crate::gpu::read_buffer(gpu, &self.state, 80)).to_vec();
        println!(
            "t={:.6} alive tris {}/{} nan verts {} bbox {:?}..{:?} max disp {:.4} | particles live {}/{} vmax {:.2} y {:.3}..{:.3} mean {:.4} vy {:.3} | spray {} region {:?}",
            self.t,
            alive.iter().filter(|&&a| a != 0).count(),
            self.n_tris,
            nan,
            lo,
            hi,
            disp,
            live,
            self.n_particles,
            vmax,
            ylo,
            yhi,
            ysum / n,
            vysum / n,
            st[16],
            &st[12..15]
        );
    }

    fn stiffness_scale(&self, t: f64) -> f64 {
        // Once the rubber has snapped back it is slack, so its (very stiff) spring rate no
        // longer matters. Softening it then lets the membrane take much larger steps.
        let t_soft = self.t_pop + 0.03;
        if t < t_soft {
            1.0
        } else {
            (-(t - t_soft) / 0.005).exp().max(1.0 / 25.0)
        }
    }

    /// Advances the simulation towards `t_target`, running at most `budget` MPM steps.
    /// Returns how many steps were issued.
    pub fn advance(&mut self, gpu: &Gpu, t_target: f64, budget: usize) -> usize {
        if !self.started {
            if t_target < 0.0 {
                self.t = t_target.max(self.t_start);
                return 0;
            }
            self.started = true;
            self.t = 0.0;
        }
        let mut issued = 0;
        while issued < budget {
            let read = self.next_read();
            if let Some((i, tr)) = read {
                if tr <= self.t + 1e-9 {
                    self.enter_water(gpu, i);
                    continue;
                }
            }
            let mut steps: Vec<(Step, u32)> = Vec::new();
            while steps.len() < MAX_BATCH && issued + steps.len() < budget {
                let dt = self.fast_dt(self.t).unwrap_or((self.dt * 1.03).min(self.dt_max));
                // Land exactly where the next bullet needs the water read back.
                let step = read.map_or(dt, |(_, tr)| dt.min(tr - self.t));
                if self.t + step > t_target + 1e-12 {
                    break;
                }
                steps.push(self.make_step(step));
                self.t += step;
                self.dt = dt;
                self.step_idx += 1;
                if read.is_some_and(|(_, tr)| self.t >= tr - 1e-9) {
                    break;
                }
            }
            if steps.is_empty() {
                break;
            }
            issued += steps.len();
            self.encode(gpu, &steps);
        }
        issued
    }

    fn make_step(&mut self, dt: f64) -> (Step, u32) {
        let t = self.t;
        let s = self.stiffness_scale(t + dt);
        let dt_m_max = self.scene.dt_mem_max / s.sqrt();
        let mut n_m = (dt / dt_m_max).ceil() as u32;
        n_m += n_m % 2;
        let n_m = n_m.max(2);
        let v_cap = (0.5 * self.dx / dt - SOUND).max(20.0);
        let mut front = [[-1e9f32; 4]; 2];
        let mut emit = [[0u32; 4]; 2];
        let mut n_emit = 0;
        for (i, sh) in self.shots.iter_mut().enumerate() {
            front[0][i] = sh.flight.front(t) as f32;
            front[1][i] = sh.flight.front(t + dt) as f32;
            for e in 0..2 {
                let Some(lip) = &sh.lips[e] else { continue };
                let total = (sh.emit_total[e] * lip.emitted(t + dt)).floor() as u64;
                let n = total.saturating_sub(sh.emitted[e]) as u32;
                sh.emitted[e] = total.max(sh.emitted[e]);
                emit[i / 2][(i % 2) * 2 + e] = n;
                n_emit += n;
            }
        }
        let step = Step {
            t: (t + dt) as f32,
            dt: dt as f32,
            dt_m: (dt / n_m as f64) as f32,
            n_particles: self.n_particles,
            gravity: [0.0, -self.params.gravity, 0.0, (SOUND * SOUND) as f32],
            grid_origin: [self.grid_origin.x, self.grid_origin.y, self.grid_origin.z, self.dx as f32],
            grid_dims: [self.dims[0], self.dims[1], self.dims[2], self.n_verts],
            front,
            emit,
            r_inf: self.scene.radius_max as f32,
            p_cav: P_ATM as f32,
            tau_vent: TAU_VENT as f32,
            v_cap: v_cap as f32,
            spacing: self.scene.spacing as f32,
            n_tris: self.n_tris,
            n_emit,
            spray_cap: SPRAY_CAP,
            n_shots: self.shots.len() as u32,
            _pad: [0; 3],
            mem: [s as f32, 0.05, 4.0, 0.08],
            spray: [self.params.mist_amount, WET_HALF, 0.0, 0.0],
            seed: [self.step_idx as u32, WET_RES, 0, 0],
            // Flying scraps have snapped back to their relaxed thickness and tumble, Cd about 1.
            misc: [0.0, (0.5 * 1.2 / (RHO_RUBBER * self.scene.thickness0)) as f32, 0.0, 0.0],
        };
        (step, n_m)
    }

    fn encode(&self, gpu: &Gpu, steps: &[(Step, u32)]) {
        let mut data = vec![0u8; steps.len() * STEP_STRIDE as usize];
        for (i, (s, _)) in steps.iter().enumerate() {
            data[i * STEP_STRIDE as usize..][..std::mem::size_of::<Step>()].copy_from_slice(bytemuck::bytes_of(s));
        }
        gpu.queue.write_buffer(&self.step_buf, 0, &data);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.mark(&mut enc, "sim");
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: None });
            let pg = self.n_particles.div_ceil(256);
            let vg = self.n_verts.div_ceil(256);
            let tg = self.n_tris.div_ceil(256);
            for (i, (s, n_m)) in steps.iter().enumerate() {
                let off = [(i as u64 * STEP_STRIDE) as u32];
                cp.set_bind_group(0, &self.bg[0], &off);
                cp.set_bind_group(1, &self.ind_bg, &[]);
                cp.set_pipeline(&self.pipes.region_prep);
                cp.dispatch_workgroups(1, 1, 1);
                cp.set_pipeline(&self.pipes.grid_clear);
                cp.dispatch_workgroups_indirect(&self.indirect, 0);
                cp.set_pipeline(&self.pipes.p2g);
                cp.dispatch_workgroups(pg, 1, 1);
                cp.set_pipeline(&self.pipes.grid_update);
                cp.dispatch_workgroups_indirect(&self.indirect, 0);
                cp.set_pipeline(&self.pipes.g2p);
                cp.dispatch_workgroups(pg, 1, 1);
                // Tearing runs through the stiff rubber within a millisecond, so it is checked
                // every other substep. Once the rubber has gone slack, once a step is enough.
                let tear_every = if s.mem[0] < 1.0 { *n_m } else { 2 };
                for m in 0..*n_m {
                    cp.set_bind_group(0, &self.bg[(m % 2) as usize], &off);
                    cp.set_pipeline(&self.pipes.mem_step);
                    cp.dispatch_workgroups(vg, 1, 1);
                    if (m + 1) % tear_every == 0 {
                        cp.set_bind_group(0, &self.bg[0], &off);
                        cp.set_pipeline(&self.pipes.mem_break);
                        cp.dispatch_workgroups(tg, 1, 1);
                    }
                }
                cp.set_bind_group(0, &self.bg[0], &off);
                let n_emit = s.n_emit;
                if n_emit > 0 {
                    cp.set_pipeline(&self.pipes.spray_emit);
                    let (gx, gy) = groups_1d(n_emit);
                    cp.dispatch_workgroups(gx, gy, 1);
                }
                cp.set_pipeline(&self.pipes.spray_update);
                cp.dispatch_workgroups_indirect(&self.indirect, 12);
            }
        }
        gpu.mark(&mut enc, "-");
        gpu.queue.submit([enc.finish()]);
    }
}
