//! Keyframe cache. The simulation runs ahead in small chunks and stores compact keyframes.
//! Playback blends between them, so any playback speed or scrub position costs the same as
//! showing a still, and the simulation itself runs at full quality.
//!
//! All GPU memory is allocated in `new`. Keyframes go into a fixed pool, and `bake` and
//! `show` never allocate. When the pool is full the timeline ends there.

use crate::gpu::{bgl, bind, groups_1d, Gpu, B};
use crate::params::SimParams;
use crate::sim::{Sim, SPRAY_CAP, WET_HALF, WET_RES};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

/// End of the timeline. Everything has landed and spread well before this.
pub const T_END: f64 = 1.0;
/// Keyframe pool, in pages so no single buffer gets huge. A keyframe never spans pages.
const PAGE_BYTES: u64 = 128 << 20;
const PAGES: usize = 16;
/// Words per keyframe header and per live spray slot, see keys.wgsl.
const KEY_HEAD: u64 = 4;
const SPRAY_WORDS: u64 = 3;

/// Keyframe spacing: 2% of the elapsed time, so the fast start gets dense keys and the fall
/// sparse ones. Linear blending over that is well below a particle spacing.
fn spacing(tau: f64) -> f64 {
    (0.02 * tau).clamp(10e-6, 4e-3)
}

/// How many keys `spacing` asks for in the first `tau` after a bullet arrives: the integral
/// of 1 / spacing.
fn keys_after(tau: f64) -> f64 {
    let (a, b) = (10e-6 / 0.02, 4e-3 / 0.02);
    tau.min(a) / 10e-6 + 50.0 * (tau.clamp(a, b) / a).ln() + (tau - b).max(0.0) / 4e-3
}

/// How many keys are still to come from `t` to the end, starting over at every arrival.
fn keys_left(t: f64, arrivals: &[f64]) -> f64 {
    let mut n = 0.0;
    for (i, &a) in arrivals.iter().enumerate() {
        let end = arrivals.get(i + 1).copied().unwrap_or(T_END);
        if end > t {
            n += keys_after(end - a) - keys_after(t.max(a) - a);
        }
    }
    n
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CacheU {
    counts: [u32; 4],
    time: [f32; 4],
    wet: [f32; 4],
    _pad: [f32; 4],
}

struct Key {
    t: f64,
    page: usize,
    offset: u64,
    size: u64,
    /// Spray slots in use at this key.
    spray: u32,
}

pub struct Cache {
    pub sim: Sim,
    keys: Vec<Key>,
    next_t: f64,
    done: bool,
    full: bool,
    pages: Vec<wgpu::Buffer>,
    /// Where the next keyframe goes: page and byte offset.
    cursor: (usize, u64),
    count_read: wgpu::Buffer,
    uni: wgpu::Buffer,
    spray_w: wgpu::Buffer,
    spray_index: wgpu::Buffer,
    index_group: wgpu::BindGroup,
    pack_bgl: wgpu::BindGroupLayout,
    show_bgl: wgpu::BindGroupLayout,
    pack: [wgpu::ComputePipeline; 4],
    index: [wgpu::ComputePipeline; 2],
    show: [wgpu::ComputePipeline; 6],

    /// What the renderer reads, rebuilt by `show`.
    pub particles: wgpu::Buffer,
    pub spray: wgpu::Buffer,
    pub mem_pos: wgpu::Buffer,
    pub alive: wgpu::Buffer,
    pub wet: wgpu::Buffer,
    pub shown: f64,
    /// Changes whenever the shown state does.
    pub shown_id: u64,
    /// Spray slots that can be in use in the shown state.
    pub shown_spray: u32,
}

impl Cache {
    pub fn new(gpu: &Gpu, params: &SimParams) -> Self {
        let sim = Sim::new(gpu, params);
        use B::*;
        let cs = wgpu::ShaderStages::COMPUTE;
        let pack_bgl = bgl(gpu, "pack", &[Uniform, Ro, Ro, Ro, Ro, Rw, Rw, Ro, Rw], cs);
        let show_bgl = bgl(gpu, "show", &[Uniform, Ro, Ro, Ro, Ro, Ro, Rw, Rw, Rw, Rw, Rw], cs);
        let common = include_str!("../shaders/common.wgsl");
        let keys = include_str!("../shaders/keys.wgsl");
        let pm = gpu.shader("pack", &[common, keys, include_str!("../shaders/pack.wgsl")]);
        let sm = gpu.shader("show", &[common, keys, include_str!("../shaders/show.wgsl")]);
        let pl = gpu.layout(&[&pack_bgl]);
        let sl = gpu.layout(&[&show_bgl]);
        let pack = ["pack_water", "pack_mem", "pack_alive", "pack_spray"].map(|e| gpu.compute(&pl, &pm, e));
        let index = ["index_bits", "index_scan"].map(|e| gpu.compute(&pl, &pm, e));
        let pages: Vec<wgpu::Buffer> = (0..PAGES).map(|_| gpu.storage("keyframe pool", PAGE_BYTES)).collect();
        let spray_w = gpu.storage("spray weights", SPRAY_CAP as u64 * 8);
        let spray_index = gpu.storage("spray index", (4 + 2 * SPRAY_CAP.div_ceil(32) as u64) * 4);
        let uni = gpu.buffer("cache", std::mem::size_of::<CacheU>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let index_group = index_group(gpu, &pack_bgl, &sim, &uni, &pages[0], &spray_w, &spray_index);
        let show = ["show_water", "show_mem", "show_alive", "show_spray", "show_wet_drops", "show_wet_touch"]
            .map(|e| gpu.compute(&sl, &sm, e));
        let [particles, spray, mem_pos, alive, wet] = shown_buffers(gpu, &sim);
        let mut c = Self {
            keys: Vec::new(),
            next_t: 0.0,
            done: false,
            full: false,
            pages,
            cursor: (0, 0),
            count_read: gpu.buffer("spray counts", 8, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            uni,
            spray_w,
            spray_index,
            index_group,
            pack_bgl,
            show_bgl,
            pack,
            index,
            show,
            particles,
            spray,
            mem_pos,
            alive,
            wet,
            shown: f64::NAN,
            shown_id: 0,
            shown_spray: 0,
            sim,
        };
        c.restart(gpu);
        c
    }

    /// Starts over with new parameters, reusing the pool.
    pub fn rebuild(&mut self, gpu: &Gpu, params: &SimParams) {
        self.sim = Sim::new(gpu, params);
        [self.particles, self.spray, self.mem_pos, self.alive, self.wet] = shown_buffers(gpu, &self.sim);
        self.index_group = index_group(gpu, &self.pack_bgl, &self.sim, &self.uni, &self.pages[0], &self.spray_w, &self.spray_index);
        self.restart(gpu);
    }

    fn restart(&mut self, gpu: &Gpu) {
        let largest_key = key_bytes(&self.sim, SPRAY_CAP, SPRAY_CAP);
        assert!(largest_key <= PAGE_BYTES, "a keyframe ({largest_key} bytes) must fit in one pool page");
        self.sim.advance(gpu, 0.0, 0);
        self.keys.clear();
        self.cursor = (0, 0);
        self.next_t = 0.0;
        self.done = false;
        self.full = false;
        self.shown = f64::NAN;
        self.shown_id += 1;
        self.store_key(gpu);
    }

    /// Time of the newest keyframe.
    pub fn end(&self) -> f64 {
        self.keys.last().map_or(0.0, |k| k.t)
    }

    pub fn done(&self) -> bool {
        self.done
    }

    /// True when the pool ran out before the end of the timeline.
    pub fn full(&self) -> bool {
        self.full
    }

    pub fn used_bytes(&self) -> u64 {
        self.keys.iter().map(|k| k.size).sum()
    }

    pub fn pool_bytes(&self) -> u64 {
        PAGE_BYTES * PAGES as u64
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    fn write_uniform(&self, gpu: &Gpu, time: [f32; 4], tb: f32) {
        let s = &self.sim;
        let u = CacheU {
            counts: [s.n_particles, s.n_verts, s.n_tris, SPRAY_CAP],
            time,
            wet: [WET_HALF, WET_RES as f32, tb, 0.0],
            _pad: [0.0; 4],
        };
        gpu.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&u));
    }

    fn key_binding(&self, k: &Key) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.pages[k.page], offset: k.offset, size: NonZeroU64::new(k.size) })
    }

    /// Runs up to `max_steps` simulation steps, storing keyframes on the way.
    pub fn bake(&mut self, gpu: &Gpu, max_steps: usize) -> usize {
        let mut steps = 0;
        while !self.done && steps < max_steps {
            let left = max_steps - steps;
            let n = self.sim.advance(gpu, self.next_t, left);
            steps += n;
            if n == left {
                continue;
            }
            if self.sim.t > self.end() {
                self.store_key(gpu);
            } else {
                // The next step is longer than the keyframe spacing: take it anyway, the
                // keyframe lands right after.
                steps += self.sim.advance(gpu, f64::INFINITY, 1);
            }
        }
        steps
    }

    /// Indexes the live spray and waits for the slot and live counts, so the keyframe gets
    /// exactly the space it needs.
    fn index_spray(&self, gpu: &Gpu) -> (u32, u32) {
        self.write_uniform(gpu, [0.0; 4], 0.0);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.mark(&mut enc, "pack");
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_bind_group(0, &self.index_group, &[]);
            let (gx, gy) = groups_1d(SPRAY_CAP.div_ceil(32));
            cp.set_pipeline(&self.index[0]);
            cp.dispatch_workgroups(gx, gy, 1);
            cp.set_pipeline(&self.index[1]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        gpu.mark(&mut enc, "-");
        enc.copy_buffer_to_buffer(&self.spray_index, 0, &self.count_read, 0, 8);
        gpu.queue.submit([enc.finish()]);
        let slice = self.count_read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let c: [u32; 2] = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[..2].try_into().unwrap();
        self.count_read.unmap();
        (c[0], c[1])
    }

    fn store_key(&mut self, gpu: &Gpu) {
        let (count, live) = self.index_spray(gpu);
        let size = key_bytes(&self.sim, count, live);
        if self.cursor.1 + size > PAGE_BYTES {
            self.cursor = (self.cursor.0 + 1, 0);
        }
        if self.cursor.0 >= self.pages.len() {
            self.full = true;
            self.done = true;
            return;
        }
        let key = Key { t: self.sim.t, page: self.cursor.0, offset: self.cursor.1, size, spray: count };
        gpu.queue.write_buffer(&self.pages[key.page], key.offset, bytemuck::cast_slice(&[count, live, 0, 0]));
        let s = &self.sim;
        let g = bind(
            gpu,
            &self.pack_bgl,
            &[
                self.uni.as_entire_binding(),
                s.particles.as_entire_binding(),
                s.mem_pos.as_entire_binding(),
                s.alive.as_entire_binding(),
                s.spray.as_entire_binding(),
                self.key_binding(&key),
                self.spray_w.as_entire_binding(),
                s.state.as_entire_binding(),
                self.spray_index.as_entire_binding(),
            ],
        );
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.mark(&mut enc, "pack");
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_bind_group(0, &g, &[]);
            let (pgx, pgy) = groups_1d(s.n_particles);
            let (sgx, sgy) = groups_1d(count);
            cp.set_pipeline(&self.pack[0]);
            cp.dispatch_workgroups(pgx, pgy, 1);
            cp.set_pipeline(&self.pack[1]);
            cp.dispatch_workgroups(s.n_verts.div_ceil(256), 1, 1);
            cp.set_pipeline(&self.pack[2]);
            cp.dispatch_workgroups(s.n_tris.div_ceil(32 * 256), 1, 1);
            cp.set_pipeline(&self.pack[3]);
            cp.dispatch_workgroups(sgx, sgy, 1);
        }
        gpu.mark(&mut enc, "-");
        gpu.queue.submit([enc.finish()]);
        self.cursor.1 += size;
        let t = key.t;
        self.keys.push(key);
        if self.next_t >= T_END {
            self.done = true;
        } else {
            // Fast bullets throw far more spray, and at the usual spacing their keys would fill
            // the pool long before the end. Stretching the spacing by however much the rest
            // would overrun keeps the whole timeline, and costs nothing when it fits.
            let free = self.pool_bytes().saturating_sub(self.used_bytes()).max(1);
            let mut arrivals: Vec<f64> = self.sim.shots.iter().map(|s| s.flight.t_hit.max(0.0)).chain([0.0]).collect();
            arrivals.sort_by(f64::total_cmp);
            arrivals.dedup();
            let last = arrivals.iter().rev().find(|&&a| a <= t).copied().unwrap_or(0.0);
            let next = arrivals.iter().find(|&&a| a > t).copied().unwrap_or(T_END);
            let overrun = 1.2 * keys_left(t, &arrivals) * size as f64 / free as f64;
            self.next_t = (t + spacing(t - last) * overrun.max(1.0)).min(next).min(T_END);
        }
    }

    /// Fills the shown state for time `t`, clamped to what has been simulated. Returns false
    /// when that is already what's shown.
    pub fn show(&mut self, gpu: &Gpu, t: f64) -> bool {
        let t = t.clamp(self.sim.t_start, self.end());
        if t == self.shown {
            return false;
        }
        let b = self.keys.partition_point(|k| k.t <= t).clamp(1, self.keys.len());
        let a = b - 1;
        let b = b.min(self.keys.len() - 1);
        let (ka, kb) = (&self.keys[a], &self.keys[b]);
        let spray = ka.spray.max(kb.spray);
        let (f, inv) = if b > a { (((t - ka.t) / (kb.t - ka.t)).clamp(0.0, 1.0), 1.0 / (kb.t - ka.t)) } else { (0.0, 0.0) };
        self.write_uniform(gpu, [t as f32, f as f32, inv as f32, ka.t as f32], kb.t as f32);
        let g = bind(
            gpu,
            &self.show_bgl,
            &[
                self.uni.as_entire_binding(),
                self.key_binding(ka),
                self.key_binding(kb),
                self.spray_w.as_entire_binding(),
                self.sim.landed.as_entire_binding(),
                self.sim.touch.as_entire_binding(),
                self.particles.as_entire_binding(),
                self.spray.as_entire_binding(),
                self.mem_pos.as_entire_binding(),
                self.alive.as_entire_binding(),
                self.wet.as_entire_binding(),
            ],
        );
        let s = &self.sim;
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.mark(&mut enc, "show");
        enc.clear_buffer(&self.wet, 0, None);
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_bind_group(0, &g, &[]);
            let (pgx, pgy) = groups_1d(s.n_particles);
            let (sgx, sgy) = groups_1d(spray);
            let (wgx, wgy) = groups_1d(WET_RES * WET_RES);
            for (pipe, (x, y)) in self.show.iter().zip([
                (pgx, pgy),
                (s.n_verts.div_ceil(256), 1),
                (s.n_tris.div_ceil(256), 1),
                (sgx, sgy),
                (sgx, sgy),
                (wgx, wgy),
            ]) {
                cp.set_pipeline(pipe);
                cp.dispatch_workgroups(x, y, 1);
            }
        }
        gpu.mark(&mut enc, "-");
        gpu.queue.submit([enc.finish()]);
        self.shown = t;
        self.shown_id += 1;
        self.shown_spray = spray;
        true
    }
}

fn shown_buffers(gpu: &Gpu, sim: &Sim) -> [wgpu::Buffer; 5] {
    [
        gpu.storage("shown particles", sim.n_particles as u64 * 32),
        gpu.storage("shown spray", SPRAY_CAP as u64 * 32),
        // Also the vertex input of the membrane's ray tracing acceleration structure.
        gpu.buffer(
            "shown membrane",
            sim.n_verts as u64 * 16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
        ),
        gpu.storage("shown alive", sim.n_tris as u64 * 4),
        gpu.storage("shown wet", (WET_RES * WET_RES) as u64 * 4),
    ]
}

/// Bytes of one keyframe with `spray` slots of which `live` are in use, rounded up to the
/// storage binding alignment.
fn key_bytes(sim: &Sim, spray: u32, live: u32) -> u64 {
    let words = KEY_HEAD
        + 2 * (sim.n_particles + sim.n_verts) as u64
        + (sim.n_tris as u64).div_ceil(32)
        + 2 * spray.div_ceil(32) as u64
        + SPRAY_WORDS * live as u64;
    (words * 4).div_ceil(256) * 256
}

/// Bindings for building the spray index. The keyframe slot is unused there, so any page
/// fills it.
fn index_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    sim: &Sim,
    uni: &wgpu::Buffer,
    page: &wgpu::Buffer,
    spray_w: &wgpu::Buffer,
    spray_index: &wgpu::Buffer,
) -> wgpu::BindGroup {
    bind(
        gpu,
        layout,
        &[
            uni.as_entire_binding(),
            sim.particles.as_entire_binding(),
            sim.mem_pos.as_entire_binding(),
            sim.alive.as_entire_binding(),
            sim.spray.as_entire_binding(),
            wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: page, offset: 0, size: NonZeroU64::new(256) }),
            spray_w.as_entire_binding(),
            sim.state.as_entire_binding(),
            spray_index.as_entire_binding(),
        ],
    )
}
