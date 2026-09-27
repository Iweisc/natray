//! The simulation. Everything that moves is stepped on the GPU, which also picks each time
//! step from the flow, so the CPU only says how far to go and reads back where it got to.

use crate::balloon::{self, Balloon};
use crate::bullet::{self, GpuBullet, PoseEntry, LOG_CAP, MAX_PATCHES, N_SEC, OUTLINE};
use crate::flight::{Flight, LOOKAHEAD};
use crate::gpu::{bgl, bind, Gpu, B};
use crate::lip::Lip;
use crate::params::{SimParams, MAX_SHOTS};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

pub const SPRAY_CAP: u32 = 1_200_000;
pub const WET_RES: u32 = 1024;
pub const WET_HALF: f32 = 1.2;
/// Speed of sound in water.
const SOUND: f32 = 1480.0;
const CFL: f32 = 0.8;
const DT_MAX: f32 = 1e-3;
const SOLVE_TOL: f32 = 1e-5;
const MAX_ITERS: u32 = 4096;
const MAX_BATCH: usize = 256;
/// Longest a batch of steps may hold the GPU. The desktop has to wait behind it.
const BATCH_MS: f64 = 4.0;
const MAX_SUBSTEPS: u32 = 256;
/// Rubber on the ground.
const FRICTION: f32 = 0.8;
/// Per-cell fields, see sim_defs.wgsl.
const ACC_FIELDS: u64 = 19;
const CELL_FIELDS: u64 = 11;
/// See membrane.wgsl.
const LAYER_MODES: u64 = 6;
/// See air.wgsl.
const AIR_FIELDS: u64 = 5;
/// The per-cell kernels run a workgroup per block along one dispatch dimension.
const MAX_BLOCKS: u32 = 65535;
/// Dispatch argument slots in bytes, see fluid.wgsl.
const IND_BLOCKS: u64 = 0;
const IND_PARTICLES: u64 = 12;
const IND_SPRAY: u64 = 24;
const IND_CLEAR: u64 = 36;
const IND_ONE: u64 = 48;
const IND_VERTS: u64 = 60;
const IND_KEEP: u64 = 72;
const IND_BULLETS: u64 = 84;
const SOLVE_BLOCKS: u64 = 0;
const SOLVE_ONE: u64 = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin: [f32; 4],
    dims: [u32; 4],
    bdims: [u32; 4],
    gravity: [f32; 4],
    fluid: [f32; 4],
    wet: [f32; 4],
    limits: [u32; 4],
    solve: [f32; 4],
    mem: [u32; 4],
    rubber: [f32; 4],
    rubber2: [f32; 4],
    knot: [f32; 4],
}

/// See State in sim_defs.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct State {
    t_ns: u32,
    dt_ns: u32,
    step: u32,
    n_blocks: u32,
    max_speed: f32,
    n_blocks_prev: u32,
    overflow: u32,
    solve_iters: u32,
    rz: f32,
    alpha: f32,
    b_norm: f32,
    r_norm: f32,
    free_dt_ns: u32,
    iters_max: u32,
    unconverged: u32,
    beta: f32,
    spray_count: u32,
    mem_dt: f32,
    cap_dt: f32,
    mem_dt_seen: f32,
    dt_flow: f32,
    air_dt: f32,
    air_dt_seen: f32,
    mem_ticket: u32,
}

/// Enough substeps, with a little to spare, that a step of `dt` isn't cut short by them.
/// Even, so a step ends with the rubber or the air back in its first buffers.
fn substeps_for(dt: f32, sub_dt: f32) -> u32 {
    let n = (1.1 * dt / sub_dt).ceil();
    if n.is_finite() {
        (n as u32).clamp(2, MAX_SUBSTEPS).next_multiple_of(2)
    } else {
        2
    }
}

pub struct ShotRun {
    pub flight: Flight,
    /// Entry and exit hole, for the renderer's crowns. Bullets don't meet the water yet, so
    /// these stay empty.
    pub lips: [Option<Lip>; 2],
}

/// Where a kernel's workgroup count comes from: one workgroup, or a slot in the step's or the
/// solver's dispatch arguments.
#[derive(Clone, Copy)]
enum Size {
    One,
    Step(u64),
    Solve(u64),
}

#[derive(Clone, Copy)]
enum Args {
    Step,
    Solve,
}

struct Kernel {
    name: &'static str,
    pipe: wgpu::ComputePipeline,
    size: Size,
    /// The dispatch arguments it writes, which it then sees as group 1.
    writes: Option<Args>,
    /// Whether it reads the rubber and the air from their second buffers.
    swap: bool,
}

/// One step's kernels in order. The solver iteration and the substep repeat as often as
/// encoded.
struct Plan {
    head: Vec<Kernel>,
    iteration: Vec<Kernel>,
    after_solve: Vec<Kernel>,
    substep: Vec<Kernel>,
    tail: Vec<Kernel>,
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
    pub mem_pos: wgpu::Buffer,
    mem_vel: wgpu::Buffer,
    mem_water: wgpu::Buffer,
    air: wgpu::Buffer,
    cf: wgpu::Buffer,
    bullets: wgpu::Buffer,
    pose_log: wgpu::Buffer,
    pose_read: wgpu::Buffer,
    pub tris: wgpu::Buffer,
    pub alive: wgpu::Buffer,
    pub vtri_off: wgpu::Buffer,
    pub vtri: wgpu::Buffer,
    pub spray: wgpu::Buffer,
    pub touch: wgpu::Buffer,
    pub landed: wgpu::Buffer,
    uniform: wgpu::Buffer,
    step_args: wgpu::Buffer,
    solve_args: wgpu::Buffer,
    state_read: wgpu::Buffer,
    bg: wgpu::BindGroup,
    /// The same with the rubber's two position and velocity buffers the other way round.
    bg_swap: wgpu::BindGroup,
    step_bg: wgpu::BindGroup,
    solve_bg: wgpu::BindGroup,
    plan: Plan,
    u: Params,
    /// As of the last read back.
    last: State,
    /// Solver iterations encoded per step, sized from what recent steps needed.
    iters: u32,
    unconverged: u32,
    /// Rubber and air substeps per step, even, sized so they rarely limit the step.
    substeps: u32,
    air_substeps: u32,
    /// Wall time of a step in the last batch.
    step_ms: f64,

    pub t: f64,
    pub t_start: f64,
    /// When the simulation's clock starts: the first bullet's handover, or the first impact.
    t0: f64,
    started: bool,
    pub shots: Vec<ShotRun>,
}

impl Sim {
    pub fn new(gpu: &Gpu, params: &SimParams) -> Self {
        let scene = balloon::build(params);
        let n_particles = scene.particles.len() as u32;
        let n_verts = scene.mem_pos.len() as u32;
        let n_tris = scene.tris.len() as u32;
        let dx = 2.0 * scene.spacing;

        // The grid is sparse, so the domain can take in everything the water could reach and
        // costs only a pointer per 4x4x4 block. The ground at y = 0 is a cell face.
        let top = scene.bottom_y + scene.profile.z_top() + 0.6;
        let nxz = ((2.0 * WET_HALF as f64 / dx).ceil() as u32).next_multiple_of(4);
        let ny = ((top / dx).ceil() as u32 + 3).next_multiple_of(4);
        let dims = [nxz, ny, nxz];
        let half = (nxz as f64 * 0.5 * dx) as f32;
        let grid_origin = Vec3::new(-half, (-3.0 * dx) as f32, -half);
        let bdims = dims.map(|d| d / 4);
        let slots = bdims.iter().map(|&b| b as u64).product::<u64>();
        let cells = MAX_BLOCKS as u64 * 64;

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
        // Nothing limits the first step's length until the rubber has measured itself. The
        // air starts still, at the stable substep air.wgsl works out for that.
        let c_air = (1.4f32 * 101325.0 / 1.204).sqrt();
        let state0 = State { mem_dt: f32::MAX, cap_dt: f32::MAX, air_dt: 0.9 * dx as f32 / (3.0 * c_air), ..State::zeroed() };
        let state = gpu.storage_init("sim state", bytemuck::bytes_of(&state0));
        let block_ptr = gpu.storage_init("block pointers", bytemuck::cast_slice(&vec![u32::MAX; slots as usize]));
        let slot_count = gpu.storage("block particle counts", slots * 4);
        let block_list = gpu.storage("block list", MAX_BLOCKS as u64 * 4);
        let block_nb = gpu.storage("block neighbours", MAX_BLOCKS as u64 * 27 * 4);
        let block_start = gpu.storage("block particle starts", (MAX_BLOCKS as u64 + 1) * 4);
        let sorted = gpu.storage("particles by block", n_particles as u64 * 4);
        let acc = gpu.storage("grid transfer", ACC_FIELDS * cells * 4);
        let cf = gpu.storage("grid fields", CELL_FIELDS * cells * 4);
        let partial = gpu.storage("solver sums", MAX_BLOCKS as u64 * 16);
        let spray = gpu.storage("spray", SPRAY_CAP as u64 * 48);
        let touch = gpu.storage_init("ground touch", bytemuck::cast_slice(&vec![u32::MAX; (WET_RES * WET_RES) as usize]));
        let landed = gpu.storage("landed", SPRAY_CAP as u64 * 16);
        let args_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT;
        let step_args = gpu.buffer("step dispatches", 96, args_usage);
        let solve_args = gpu.buffer("solver dispatches", 32, args_usage);
        let uniform = gpu.buffer("sim params", std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let state_read = gpu.buffer("sim state read", std::mem::size_of::<State>() as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

        let mem_pos = gpu.storage_init("mem pos", bytemuck::cast_slice(&scene.mem_pos));
        let mem_pos_b = gpu.storage_init("mem pos b", bytemuck::cast_slice(&scene.mem_pos));
        let mem_vel = gpu.storage("mem vel", n_verts as u64 * 16);
        let mem_vel_b = gpu.storage("mem vel b", n_verts as u64 * 16);
        let mem_mass = gpu.storage_init("mem mass", bytemuck::cast_slice(&scene.mem_mass));
        let mem_layer = gpu.storage("mem water layer", n_verts as u64 * LAYER_MODES * 16);
        let mem_push = gpu.storage("mem push", n_verts as u64 * 8);
        // The rubber starts on the water everywhere.
        let mem_water = gpu.storage_init("mem water", bytemuck::cast_slice(&vec![[0f32, 0.0, 0.0, 1.0]; n_verts as usize]));
        let tris = gpu.storage_init("tris", bytemuck::cast_slice(&scene.tris));
        let alive = gpu.storage_init("alive", bytemuck::cast_slice(&vec![1u32; n_tris as usize]));
        let vtri_off = gpu.storage_init("vtri off", bytemuck::cast_slice(&scene.vtri_off));
        let vtri = gpu.storage_init("vtri", bytemuck::cast_slice(&scene.vtri));
        let air = gpu.storage("air", AIR_FIELDS * cells * 4);
        let air_b = gpu.storage("air b", AIR_FIELDS * cells * 4);
        let prev_ptr = gpu.storage_init("previous block pointers", bytemuck::cast_slice(&vec![u32::MAX; slots as usize]));
        let block_list_prev = gpu.storage("previous block list", MAX_BLOCKS as u64 * 4);
        let block_active = gpu.storage("active air blocks", MAX_BLOCKS as u64 * 4);

        let first = params.shots.iter().map(|s| s.delay_ms).fold(f32::MAX, f32::min);
        let shots: Vec<ShotRun> = params
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
                ShotRun { flight, lips: [None, None] }
            })
            .collect();
        let t_start = shots.iter().map(|s| s.flight.t_read - (0.3 - LOOKAHEAD) / s.flight.report.v_impact).fold(0.0, f64::min);
        let t0 = shots.iter().map(|s| s.flight.t_read).fold(0.0, f64::min);

        let mut bullets_init = vec![GpuBullet::zeroed(); MAX_SHOTS];
        let mut patches = vec![[0f32; 8]; MAX_SHOTS * MAX_PATCHES];
        let mut outlines = vec![[0f32; 2]; MAX_SHOTS * OUTLINE];
        for (k, s) in shots.iter().enumerate() {
            let surface = bullet::surface(&s.flight.body);
            patches[k * MAX_PATCHES..k * MAX_PATCHES + surface.len()].copy_from_slice(bytemuck::cast_slice(&surface));
            outlines[k * OUTLINE..(k + 1) * OUTLINE].copy_from_slice(&bullet::gpu_outline(&s.flight.body));
            bullets_init[k] = bullet::initial(&s.flight, ((s.flight.t_read - t0) * 1e9).round() as u32, surface.len());
        }
        let bullets = gpu.storage_init("bullets", bytemuck::cast_slice(&bullets_init));
        let patches = gpu.storage_init("bullet patches", bytemuck::cast_slice(&patches));
        let outlines = gpu.storage_init("bullet outlines", bytemuck::cast_slice(&outlines));
        let bullet_hit = gpu.storage("rubber on bullets", (MAX_SHOTS * 13 * 4) as u64);
        let sections = gpu.storage("cavity sections", (MAX_SHOTS * N_SEC * 48) as u64);
        let log_size = (MAX_SHOTS * LOG_CAP * std::mem::size_of::<PoseEntry>()) as u64;
        let pose_log = gpu.buffer("bullet poses", log_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let pose_read = gpu.buffer("bullet poses read", log_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

        use B::*;
        let cs = wgpu::ShaderStages::COMPUTE;
        let mut entries = vec![Uniform];
        entries.resize(38, Rw);
        let layout = bgl(gpu, "sim", &entries, cs);
        let sim_bg = |mem: [&wgpu::Buffer; 4], gas: [&wgpu::Buffer; 2]| {
            bind(
                gpu,
                &layout,
                &[
                    uniform.as_entire_binding(),
                    state.as_entire_binding(),
                    particles.as_entire_binding(),
                    block_ptr.as_entire_binding(),
                    slot_count.as_entire_binding(),
                    block_list.as_entire_binding(),
                    block_nb.as_entire_binding(),
                    block_start.as_entire_binding(),
                    sorted.as_entire_binding(),
                    acc.as_entire_binding(),
                    cf.as_entire_binding(),
                    partial.as_entire_binding(),
                    spray.as_entire_binding(),
                    touch.as_entire_binding(),
                    landed.as_entire_binding(),
                    mem[0].as_entire_binding(),
                    mem[1].as_entire_binding(),
                    mem[2].as_entire_binding(),
                    mem[3].as_entire_binding(),
                    tris.as_entire_binding(),
                    vtri_off.as_entire_binding(),
                    vtri.as_entire_binding(),
                    mem_mass.as_entire_binding(),
                    mem_water.as_entire_binding(),
                    alive.as_entire_binding(),
                    mem_layer.as_entire_binding(),
                    mem_push.as_entire_binding(),
                    gas[0].as_entire_binding(),
                    gas[1].as_entire_binding(),
                    prev_ptr.as_entire_binding(),
                    block_list_prev.as_entire_binding(),
                    block_active.as_entire_binding(),
                    bullets.as_entire_binding(),
                    patches.as_entire_binding(),
                    sections.as_entire_binding(),
                    pose_log.as_entire_binding(),
                    outlines.as_entire_binding(),
                    bullet_hit.as_entire_binding(),
                ],
            )
        };
        let bg = sim_bg([&mem_pos, &mem_vel, &mem_pos_b, &mem_vel_b], [&air, &air_b]);
        let bg_swap = sim_bg([&mem_pos_b, &mem_vel_b, &mem_pos, &mem_vel], [&air_b, &air]);
        let args_layout = bgl(gpu, "sim dispatches", &[Rw], cs);
        let step_bg = bind(gpu, &args_layout, &[step_args.as_entire_binding()]);
        let solve_bg = bind(gpu, &args_layout, &[solve_args.as_entire_binding()]);
        let module = gpu.shader(
            "fluid",
            &[include_str!("../shaders/common.wgsl"), include_str!("../shaders/sim_defs.wgsl"), include_str!("../shaders/fluid.wgsl"), include_str!("../shaders/membrane.wgsl"), include_str!("../shaders/air.wgsl"), include_str!("../shaders/bullet.wgsl")],
        );
        // Only kernels that write dispatch arguments see them as storage. The rest are
        // dispatched from them, and a buffer can't be both in one dispatch.
        let pl = gpu.layout(&[&layout]);
        let pl_args = gpu.layout(&[&layout, &args_layout]);
        let k = |name: &'static str, size: Size, writes: Option<Args>| Kernel {
            name,
            pipe: gpu.compute(if writes.is_some() { &pl_args } else { &pl }, &module, name),
            size,
            writes,
            swap: false,
        };
        let swapped = |name, size| Kernel { swap: true, ..k(name, size, None) };
        use Size::*;
        let plan = Plan {
            head: vec![
                k("begin_step", One, Some(Args::Step)),
                k("clear_blocks", Step(IND_CLEAR), None),
                k("keep_blocks", Step(IND_KEEP), None),
                k("mark_blocks", Step(IND_PARTICLES), None),
                k("mark_rubber", Step(IND_VERTS), None),
                k("blocks_ready", One, Some(Args::Step)),
                k("scan_blocks", Step(IND_ONE), None),
                k("sort_particles", Step(IND_PARTICLES), None),
                k("p2g", Step(IND_BLOCKS), None),
                // Air goes from the air buffer into the second one in the new blocks, and comes
                // back once the cells' types are known.
                k("air_carry", Step(IND_BLOCKS), None),
                k("mem_splat", Step(IND_VERTS), None),
                k("grid_prep", Step(IND_BLOCKS), None),
                swapped("air_fill", Step(IND_BLOCKS)),
                k("bullet_step", Step(IND_BULLETS), None),
                k("bullet_apply", Step(IND_BLOCKS), None),
                k("solve_init", Step(IND_BLOCKS), None),
                k("solve_start", Step(IND_ONE), Some(Args::Solve)),
            ],
            iteration: vec![
                k("solve_q", Solve(SOLVE_BLOCKS), None),
                k("solve_alpha", Solve(SOLVE_ONE), None),
                k("solve_update", Solve(SOLVE_BLOCKS), None),
                k("solve_beta", Step(IND_ONE), Some(Args::Solve)),
            ],
            after_solve: vec![
                k("project", Step(IND_BLOCKS), None),
                k("extend_1", Step(IND_BLOCKS), None),
                k("extend_2", Step(IND_BLOCKS), None),
                k("mem_sample", Step(IND_VERTS), None),
            ],
            substep: vec![k("mem_substep", Step(IND_VERTS), None), k("air_substep", Step(IND_BLOCKS), None)],
            tail: vec![k("spray_update", Step(IND_SPRAY), None), k("g2p", Step(IND_BLOCKS), None)],
        };

        let u = Params {
            origin: grid_origin.extend(dx as f32).into(),
            dims: [dims[0], dims[1], dims[2], n_particles],
            bdims: [bdims[0], bdims[1], bdims[2], MAX_BLOCKS],
            gravity: [0.0, -params.gravity, 0.0, SOUND],
            fluid: [balloon::RHO_WATER as f32, CFL, DT_MAX, scene.spacing as f32],
            wet: [WET_HALF, 0.0, 0.0, 0.0],
            limits: [WET_RES, SPRAY_CAP, 0, 0],
            solve: [SOLVE_TOL, 0.0, 0.0, 0.0],
            mem: [n_verts, 2, 2, shots.len() as u32],
            rubber: [scene.latex.mu[0] as f32, scene.latex.mu[1] as f32, scene.latex.mu[2] as f32, scene.latex.thickness as f32],
            rubber2: [FRICTION, params.rubber_break_stretch, 0.0, 0.0],
            knot: [0.0, (scene.bottom_y + scene.profile.z_neck) as f32, 0.0, scene.profile.neck_r as f32],
        };


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
            mem_pos,
            mem_vel,
            mem_water,
            air,
            cf,
            bullets,
            pose_log,
            pose_read,
            tris,
            alive,
            vtri_off,
            vtri,
            spray,
            touch,
            landed,
            uniform,
            step_args,
            solve_args,
            state_read,
            bg,
            bg_swap,
            step_bg,
            solve_bg,
            plan,
            u,
            last: State::zeroed(),
            iters: 64,
            unconverged: 0,
            substeps: 2,
            air_substeps: 2,
            step_ms: 0.0,
            t: t_start,
            t_start,
            t0,
            started: false,
            shots,
            scene,
        };
        sim.measure_rubber(gpu);
        sim
    }

    /// Advances the simulation to `t_target`, running at most `budget` steps. Returns how
    /// many it took.
    pub fn advance(&mut self, gpu: &Gpu, t_target: f64, budget: usize) -> usize {
        if !self.started {
            if t_target < self.t0 {
                self.t = t_target.max(self.t_start);
                return 0;
            }
            self.started = true;
            self.t = self.t0;
        }
        // Nanoseconds in a u32 reach 4.29 s, well past the end of the timeline.
        let stop = ((t_target - self.t0).min(4.0) * 1e9).round() as u32;
        let mut taken = 0;
        while taken < budget && self.last.t_ns < stop {
            // As many steps as the current one would take. The GPU stops at `stop` if they
            // grow, and the next batch carries on if they shrink.
            let dt = if self.last.free_dt_ns > 0 { self.last.free_dt_ns } else { (DT_MAX * 1e9) as u32 };
            let fit = if self.step_ms > 0.0 { (BATCH_MS / self.step_ms) as usize } else { 1 };
            let n = ((stop - self.last.t_ns).div_ceil(dt) as usize).clamp(1, (budget - taken).min(MAX_BATCH).min(fit.max(1)));
            let before = self.last.step;
            self.run(gpu, n, stop);
            taken += (self.last.step - before) as usize;
        }
        self.t = self.t0 + self.last.t_ns as f64 * 1e-9;
        taken
    }

    /// Runs a substep of no length, which only measures the rubber, so the first step knows
    /// how long it may be and how many substeps it needs.
    fn measure_rubber(&mut self, gpu: &Gpu) {
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.u));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_bind_group(0, &self.bg, &[]);
            cp.set_pipeline(&self.plan.substep[0].pipe);
            cp.dispatch_workgroups(self.n_verts.div_ceil(256), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.state, 0, &self.state_read, 0, std::mem::size_of::<State>() as u64);
        gpu.queue.submit([enc.finish()]);
        self.read_state(gpu);
        let s = self.last;
        self.substeps = substeps_for(DT_MAX.min(s.cap_dt), s.mem_dt);
        self.air_substeps = substeps_for(DT_MAX.min(s.cap_dt), s.air_dt);
    }

    fn read_state(&mut self, gpu: &Gpu) {
        let slice = self.state_read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        self.last = *bytemuck::from_bytes(&slice.get_mapped_range());
        self.state_read.unmap();
    }

    /// Each kernel of a step, and whether it reads the rubber and the air from their second
    /// buffers.
    fn kernels(&self) -> impl Iterator<Item = (&Kernel, bool)> {
        let p = &self.plan;
        let (n_mem, n_air) = (self.substeps, self.air_substeps);
        fn plain(k: &Kernel) -> (&Kernel, bool) {
            (k, k.swap)
        }
        p.head
            .iter()
            .map(plain)
            .chain((0..self.iters).flat_map(|_| &p.iteration).map(plain))
            .chain(p.after_solve.iter().map(plain))
            .chain((0..n_mem.max(n_air)).flat_map(move |i| p.substep.iter().filter(move |k| i < if k.name == "air_substep" { n_air } else { n_mem }).map(move |k| (k, i % 2 == 1))))
            .chain(p.tail.iter().map(plain))
    }

    fn dispatch(&self, cp: &mut wgpu::ComputePass, (k, swap): (&Kernel, bool)) {
        cp.set_bind_group(0, if swap { &self.bg_swap } else { &self.bg }, &[]);
        match k.writes {
            Some(Args::Step) => cp.set_bind_group(1, &self.step_bg, &[]),
            Some(Args::Solve) => cp.set_bind_group(1, &self.solve_bg, &[]),
            None => {}
        }
        cp.set_pipeline(&k.pipe);
        match k.size {
            Size::One => cp.dispatch_workgroups(1, 1, 1),
            Size::Step(slot) => cp.dispatch_workgroups_indirect(&self.step_args, slot),
            Size::Solve(slot) => cp.dispatch_workgroups_indirect(&self.solve_args, slot),
        }
    }

    /// Encodes `n` steps that stop at `stop`, runs them and reads back where they got to.
    /// With NATRAY_PROF set, each kernel gets its own timed pass instead.
    fn run(&mut self, gpu: &Gpu, n: usize, stop: u32) {
        let before = self.last.step;
        self.u.limits[2] = self.iters;
        self.u.limits[3] = stop;
        self.u.mem[1] = self.substeps;
        self.u.mem[2] = self.air_substeps;
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.u));
        // iters_max and unconverged count per batch.
        gpu.queue.write_buffer(&self.state, 13 * 4, bytemuck::cast_slice(&[0u32; 2]));
        let profile = std::env::var_os("NATRAY_PROF").is_some();
        let per_step = self.kernels().count() as u32;
        let queries = profile.then(|| {
            let qs = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * per_step * n as u32 });
            let rb = gpu.buffer("kernel times", 16 * (per_step * n as u32) as u64, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
            (qs, rb)
        });
        let started = std::time::Instant::now();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.mark(&mut enc, "sim");
        if let Some((qs, rb)) = &queries {
            let mut q = 0;
            for _ in 0..n {
                for k in self.kernels() {
                    let writes = wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(q), end_of_pass_write_index: Some(q + 1) };
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(k.0.name), timestamp_writes: Some(writes) });
                    self.dispatch(&mut cp, k);
                    q += 2;
                }
            }
            enc.resolve_query_set(qs, 0..q, rb, 0);
        } else {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: None });
            for _ in 0..n {
                for k in self.kernels() {
                    self.dispatch(&mut cp, k);
                }
            }
        }
        gpu.mark(&mut enc, "-");
        enc.copy_buffer_to_buffer(&self.state, 0, &self.state_read, 0, std::mem::size_of::<State>() as u64);
        enc.copy_buffer_to_buffer(&self.pose_log, 0, &self.pose_read, 0, self.pose_read.size());
        gpu.queue.submit([enc.finish()]);
        self.read_state(gpu);
        self.step_ms = started.elapsed().as_secs_f64() * 1e3 / n as f64;
        self.read_poses(gpu, before);
        if let Some((_, rb)) = &queries {
            self.print_profile(gpu, rb, n);
        }

        let s = &self.last;
        assert!(s.overflow == 0, "the water needs more than {MAX_BLOCKS} grid blocks");
        self.unconverged += s.unconverged;
        // An iteration costs the same whether or not the solver still needs it, so encode
        // only a little more than recent steps took, and double on any step that ran out.
        self.iters = if s.unconverged > 0 {
            (self.iters * 2).min(MAX_ITERS)
        } else if s.iters_max > 0 {
            (s.iters_max + s.iters_max / 4 + 2).clamp(8, MAX_ITERS)
        } else {
            self.iters
        };
        self.substeps = substeps_for(s.dt_flow, s.mem_dt_seen);
        self.air_substeps = substeps_for(s.dt_flow, s.air_dt_seen);
    }

    /// Hands the renderer where each bullet was at every step since step `before`.
    fn read_poses(&mut self, gpu: &Gpu, before: u32) {
        let slice = self.pose_read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let log: Vec<PoseEntry> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.pose_read.unmap();
        for (k, shot) in self.shots.iter_mut().enumerate() {
            for step in before + 1..=self.last.step {
                let e = &log[k * LOG_CAP + step as usize % LOG_CAP];
                let t_ns = e.x[3].to_bits();
                if t_ns == 0 {
                    continue;
                }
                let d = |a: [f32; 4]| glam::DVec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
                shot.flight.push_pose(self.t0 + t_ns as f64 * 1e-9, d(e.x), d(e.v), e.axis(), e.v[3] as f64);
            }
        }
    }

    fn print_profile(&self, gpu: &Gpu, rb: &wgpu::Buffer, n: usize) {
        let names: Vec<&str> = self.kernels().map(|(k, _)| k.name).collect();
        let raw = crate::gpu::read_buffer(gpu, rb, rb.size());
        let ts: &[u64] = bytemuck::cast_slice(&raw);
        let ns = gpu.queue.get_timestamp_period() as f64;
        let mut per: Vec<(&str, f64, u32)> = Vec::new();
        for (i, pair) in ts.chunks(2).enumerate() {
            let us = pair[1].wrapping_sub(pair[0]) as f64 * ns * 1e-3;
            let name = names[i % names.len()];
            match per.iter_mut().find(|p| p.0 == name) {
                Some(p) => (p.1, p.2) = (p.1 + us, p.2 + 1),
                None => per.push((name, us, 1)),
            }
        }
        let total: f64 = per.iter().map(|p| p.1).sum();
        println!("kernel time over {n} steps from step {}: {:.0} us per step", self.last.step, total / n as f64);
        for (name, us, count) in per {
            println!("  {name:15} {:8.1} us per step, {:6.2} us per dispatch", us / n as f64, us / count as f64);
        }
    }

    pub fn debug_stats(&self, gpu: &Gpu) {
        let raw = crate::gpu::read_buffer(gpu, &self.particles, self.n_particles as u64 * 80);
        let parts: Vec<&[f32; 20]> = bytemuck::cast_slice::<u8, [f32; 20]>(&raw).iter().filter(|p| p[7] > 0.5).collect();
        let n = parts.len().max(1) as f32;
        let (mut lo, mut hi, mut mean, mut vmean, mut vmax) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), Vec3::ZERO, Vec3::ZERO, 0f32);
        let mut fastest = Vec3::ZERO;
        let mut ke = 0.0f64;
        let mut cells = std::collections::HashMap::<[i32; 3], u32>::new();
        for p in &parts {
            let x = Vec3::new(p[0], p[1], p[2]);
            let v = Vec3::new(p[4], p[5], p[6]);
            lo = lo.min(x);
            hi = hi.max(x);
            mean += x / n;
            vmean += v / n;
            ke += 0.5 * v.length_squared() as f64;
            if v.length() > vmax {
                vmax = v.length();
                fastest = x;
            }
            let c = ((x - self.grid_origin) / self.dx as f32).floor().as_ivec3();
            *cells.entry(c.into()).or_default() += 1;
        }
        // Cells at least half as full as bulk water, times their volume: a volume that
        // ignores the half-empty surface cells.
        let full = cells.values().filter(|&&c| c >= 4).count() as f64;
        let s = &self.last;
        println!(
            "t={:.6} dt={}ns steps {} | water {}/{} bbox {:.3?}..{:.3?} mean {:.4?} v {:.3?} vmax {:.2} at {:.4?} volume {:.3} L | blocks {} solver {} iters (encoded {}) r {:.2e} of {:.2e}, {} unconverged steps | spray {}",
            self.t,
            s.dt_ns,
            s.step,
            parts.len(),
            self.n_particles,
            lo,
            hi,
            mean,
            vmean,
            vmax,
            fastest,
            full * self.dx.powi(3) * 1e3,
            s.n_blocks,
            s.solve_iters,
            self.iters,
            s.r_norm,
            s.b_norm,
            self.unconverged,
            s.spray_count
        );
        let raw = crate::gpu::read_buffer(gpu, &self.mem_pos, self.n_verts as u64 * 16);
        let x: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        let raw = crate::gpu::read_buffer(gpu, &self.mem_water, self.n_verts as u64 * 16);
        let w: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        let raw = crate::gpu::read_buffer(gpu, &self.alive, self.n_tris as u64 * 4);
        let alive: &[u32] = bytemuck::cast_slice(&raw);
        let raw = crate::gpu::read_buffer(gpu, &self.mem_vel, self.n_verts as u64 * 16);
        let rubber_vmax = bytemuck::cast_slice::<u8, [f32; 4]>(&raw).iter().map(|v| Vec3::from_slice(&v[..3]).length()).fold(0.0, f32::max);
        let mut volume = 0.0;
        for t in &self.scene.tris {
            let [a, b, c] = t.v.map(|i| Vec3::from_slice(&x[i as usize][..3]).as_dvec3());
            volume += a.dot(b.cross(c)) / 6.0;
        }
        let free: Vec<usize> = (0..x.len()).filter(|&i| self.scene.mem_mass[i] > 0.0).collect();
        let on = free.iter().filter(|&&i| w[i][3] >= 0.05).count();
        let fill_min = free.iter().map(|&i| w[i][3]).fold(f32::MAX, f32::min);
        let fill_mean = free.iter().map(|&i| w[i][3]).sum::<f32>() / free.len() as f32;
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in x {
            lo = lo.min(Vec3::from_slice(&p[..3]));
            hi = hi.max(Vec3::from_slice(&p[..3]));
        }
        // The most tense triangles, where the step limits come from.
        let mut worst: Vec<(f64, usize, f64, f64)> = self
            .scene
            .tris
            .iter()
            .enumerate()
            .filter(|&(ti, _)| alive[ti] != 0)
            .map(|(ti, t)| {
                let [a, b, c] = t.v.map(|i| Vec3::from_slice(&x[i as usize][..3]).as_dvec3());
                let m = t.dm_inv.map(|v| v as f64);
                let (e1, e2) = (b - a, c - a);
                let (f0, f1) = (e1 * m[0] + e2 * m[2], e1 * m[1] + e2 * m[3]);
                let (c00, c01, c11) = (f0.dot(f0), f0.dot(f1), f1.dot(f1));
                let mid = 0.5 * (c00 + c11);
                let dev = (0.25 * (c00 - c11).powi(2) + c01 * c01).sqrt();
                let (l1, l2) = ((mid + dev).sqrt(), (mid - dev).max(0.0).sqrt());
                let t = self.scene.latex.stress(l1, l2).0 * self.scene.latex.thickness / (l1 * l2).max(1e-9);
                (t, ti, l1, l2)
            })
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(tension, ti, l1, l2) in worst.iter().take(3) {
            let t = &self.scene.tris[ti];
            let held = t.v.iter().filter(|&&i| self.scene.mem_mass[i as usize] == 0.0).count();
            let cen = t.v.iter().map(|&i| Vec3::from_slice(&x[i as usize][..3])).sum::<Vec3>() / 3.0;
            println!("  tension {tension:.0} N/m stretch {l1:.2} x {l2:.3} at {:.4?} ({held} corners held), above bottom {:.4}", cen, cen.y as f64 - self.scene.bottom_y);
        }
        println!(
            "  rubber: inside {:.4} L (filled {:.4}) bbox {:.4?}..{:.4?} on water {on}/{} fill min {fill_min:.2} mean {fill_mean:.2} torn {} fastest {rubber_vmax:.1} m/s | substep {:.2}us x{} (step could be {:.1}us) capillary {:.1}us",
            volume * 1e3,
            self.scene.profile.volume * 1e3,
            lo,
            hi,
            free.len(),
            alive.iter().filter(|&&a| a == 0).count(),
            s.mem_dt_seen * 1e6,
            self.substeps,
            s.dt_flow * 1e6,
            s.cap_dt * 1e6,
        );
        // The air of the cells in use: gauge pressure range, how much of it is vapour, and
        // how fast it moves.
        let cells = s.n_blocks as u64 * 64;
        let cap = MAX_BLOCKS as u64 * 64;
        let field = |buf: &wgpu::Buffer, f: u64| -> Vec<f32> { bytemuck::cast_slice(&crate::gpu::read_buffer_at(gpu, buf, f * cap * 4, cells * 4)).to_vec() };
        let ty = field(&self.cf, 10);
        let air: Vec<Vec<f32>> = (0..AIR_FIELDS).map(|f| field(&self.air, f)).collect();
        let (mut n, mut vapour, mut plo, mut phi, mut umax) = (0, 0, f32::MAX, f32::MIN, 0f32);
        for c in 0..cells as usize {
            if ty[c] != 0.0 {
                continue;
            }
            let rho = air[0][c];
            let m = Vec3::new(air[1][c], air[2][c], air[3][c]);
            let p = 0.4 * (air[4][c] - 0.5 * m.length_squared() / rho) - 101325.0;
            n += 1;
            vapour += (p < -90e3) as u32;
            plo = plo.min(p);
            phi = phi.max(p);
            umax = umax.max(m.length() / rho);
        }
        let raw = crate::gpu::read_buffer(gpu, &self.bullets, self.bullets.size());
        for (k, b) in bytemuck::cast_slice::<u8, GpuBullet>(&raw).iter().take(self.shots.len()).enumerate() {
            let v = Vec3::from_slice(&b.v[..3]);
            let q = glam::Quat::from_array(b.q);
            let axis = q * Vec3::Z;
            let spin = axis.dot(Vec3::from_slice(&b.l[..3])) / b.shape[0];
            println!(
                "  bullet {k}: state {} at {:.4?} v {:.1} m/s yaw {:.1} deg spin {:.0} rev/s, {} sections",
                b.info[2],
                Vec3::from_slice(&b.x[..3]),
                v.length(),
                axis.angle_between(v).to_degrees(),
                spin / std::f32::consts::TAU,
                b.info[3]
            );
        }
        if std::env::var_os("NATRAY_POSES").is_some() {
            for (k, shot) in self.shots.iter().enumerate() {
                let mut next = f64::MIN;
                for w in shot.flight.poses.windows(2).filter(|w| w[1].t >= self.t0) {
                    let (a, p) = (&w[0], &w[1]);
                    if p.t >= next {
                        let turn = a.n.angle_between(p.n) / (p.t - a.t).max(1e-12);
                        println!("  pose {k}: t {:.6} x {:.4?} v {:.1} yaw {:.1} turn {turn:.0} rad/s", p.t, p.x.to_array(), p.u.length(), p.n.angle_between(p.u).to_degrees());
                        next = p.t + 25e-6;
                    }
                }
            }
        }
        let m_particle = balloon::RHO_WATER * self.scene.spacing.powi(3);
        let bullet_ke: f64 = bytemuck::cast_slice::<u8, GpuBullet>(&raw).iter().take(self.shots.len()).map(|b| 0.5 * b.x[3] as f64 * Vec3::from_slice(&b.v[..3]).length_squared() as f64).sum();
        let cavity = vapour as f64 * self.dx.powi(3);
        println!(
            "  energy: water {:.2} J, bullets {:.2} J, vapour {:.1} cm^3 (x 1 atm = {:.2} J)",
            ke * m_particle,
            bullet_ke,
            cavity * 1e6,
            cavity * 101325.0
        );
        println!("  air: {n} cells, {vapour} of them vapour, gauge {plo:.1}..{phi:.1} Pa, fastest {umax:.2} m/s | substep {:.2}us x{}", s.air_dt_seen * 1e6, self.air_substeps);
    }
}

