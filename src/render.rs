use crate::env::{Env, Ground};
use crate::gpu::{bgl, bind, groups_1d, texture, Gpu, B};
use crate::params::{LookParams, MAX_SHOTS};
use crate::cache::Cache;
use crate::lip::{P_ATM, TAU_VENT};
use crate::sim::{WET_HALF, WET_RES};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

const PHOTON_N: u32 = 1024;
/// Water splat kernel radius in particle spacings.
const KERNEL: f32 = 2.4;
const BLOOM_LEVELS: usize = 6;
/// Water brick pool, see ATLAS_B in defs.wgsl.
const POOL: u64 = 32 * 32 * 32;
/// Progressive refinement stops here; the converged image is just re-displayed.
pub const MAX_ACCUM: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameU {
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    res: [f32; 4],
    jitter: [f32; 4],
    sun_dir: [f32; 4],
    sun_col: [f32; 4],
    sky: [f32; 4],
    sh: [[f32; 4]; 9],
    knot: [f32; 4],
    rubber: [f32; 4],
    rubber2: [f32; 4],
    water: [f32; 4],
    wet: [f32; 4],
    flags: [u32; 4],
    counts: [u32; 4],
    time: [f32; 4],
    lens: [f32; 4],
    bullets: [BulletU; MAX_SHOTS],
}

/// See Bullet in defs.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BulletU {
    pos: [f32; 4],
    dir: [f32; 4],
    look: [f32; 4],
}

/// The crown sheets at the holes, see crown.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrownU {
    mouth: [[f32; 4]; 2 * MAX_SHOTS],
    axis: [[f32; 4]; 2 * MAX_SHOTS],
    lip: [[f32; 4]; 2 * MAX_SHOTS],
    misc: [f32; 4],
}

/// Crown mesh size per hole, matching NPHI and NROW in crown.wgsl.
const CROWN_VERTS: u32 = 384 * 48 * 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostU {
    rect: [f32; 4],
    res: [f32; 4],
    misc: [f32; 4],
    _pad: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

impl Camera {
    pub fn position(&self) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        self.target + Vec3::new(p.cos() * y.sin(), p.sin(), p.cos() * y.cos()) * self.dist
    }
}

struct MemAccel {
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    blas: wgpu::Blas,
    tlas: wgpu::Tlas,
}

impl MemAccel {
    fn new(gpu: &Gpu, n_verts: u32, n_tris: u32) -> Self {
        // The driver sizes these itself. This is a generous estimate for the memory ceiling.
        gpu.reserve("membrane acceleration structure", n_tris as u64 * 256);
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: n_verts,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(n_tris * 3),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD;
        let update_mode = wgpu::AccelerationStructureUpdateMode::Build;
        let blas = gpu.device.create_blas(
            &wgpu::CreateBlasDescriptor { label: Some("membrane"), flags, update_mode },
            wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
        );
        let mut tlas = gpu.device.create_tlas(&wgpu::CreateTlasDescriptor { label: Some("membrane"), max_instances: 1, flags, update_mode });
        tlas[0] = Some(wgpu::TlasInstance::new(&blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xff));
        Self { size, blas, tlas }
    }

    fn build(&self, enc: &mut wgpu::CommandEncoder, verts: &wgpu::Buffer, indices: &wgpu::Buffer) {
        let geometry = wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &self.size,
            vertex_buffer: verts,
            first_vertex: 0,
            vertex_stride: 16,
            index_buffer: Some(indices),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]);
        enc.build_acceleration_structures([&wgpu::BlasBuildEntry { blas: &self.blas, geometry }], [&self.tlas]);
    }
}

struct SizeRes {
    w: u32,
    h: u32,
    frame_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    accum: wgpu::Buffer,
    bloom: Vec<(wgpu::TextureView, (u32, u32))>,
}

struct Groups {
    prep: wgpu::BindGroup,
    scene: wgpu::BindGroup,
    caustic: wgpu::BindGroup,
    accum: wgpu::BindGroup,
    bloom: Vec<wgpu::BindGroup>,
    comp: wgpu::BindGroup,
    sprite: wgpu::BindGroup,
    crown: wgpu::BindGroup,
}

pub struct Renderer {
    pub env: Env,
    ground: Ground,
    lin_samp: wgpu::Sampler,
    env_samp: wgpu::Sampler,
    ground_samp: wgpu::Sampler,
    frame_buf: wgpu::Buffer,
    post_buf: wgpu::Buffer,
    crown_buf: wgpu::Buffer,

    rstate: wgpu::Buffer,
    wgrid: wgpu::Buffer,
    mgrid: wgpu::Buffer,
    sgrid: wgpu::Buffer,
    cmap: wgpu::Buffer,
    indirect: wgpu::Buffer,
    prep_ind: wgpu::BindGroup,
    dens_acc: wgpu::Buffer,
    dens_view: wgpu::TextureView,
    brick_range: wgpu::Buffer,
    slot_brick: wgpu::Buffer,
    vindex: wgpu::Buffer,
    n_vbricks: u32,
    mem_norm: wgpu::Buffer,
    /// Membrane triangles for the ray tracing hardware, rebuilt with the shown state.
    mem_index: wgpu::Buffer,
    mem_accel: Option<MemAccel>,
    mist_acc: wgpu::Buffer,
    mist_blur: wgpu::Buffer,
    mist_view: wgpu::TextureView,
    caustic_acc: wgpu::Buffer,
    caustic_view: wgpu::TextureView,
    sm_acc: wgpu::Buffer,
    spos: wgpu::Buffer,
    aniso: wgpu::Buffer,
    /// Sim step and sun direction the acceleration structures were last built for.
    prep_key: Option<(u64, [u32; 3])>,

    size: Option<SizeRes>,
    groups: Option<Groups>,

    prep_bgl: wgpu::BindGroupLayout,
    scene_bgl: wgpu::BindGroupLayout,
    caustic_bgl: wgpu::BindGroupLayout,
    accum_bgl: wgpu::BindGroupLayout,
    bloom_bgl: wgpu::BindGroupLayout,
    comp_bgl: wgpu::BindGroupLayout,
    sprite_bgl: wgpu::BindGroupLayout,
    crown_bgl: wgpu::BindGroupLayout,

    prep: Vec<(&'static str, wgpu::ComputePipeline)>,
    trace: wgpu::ComputePipeline,
    photons: wgpu::ComputePipeline,
    caustic: wgpu::ComputePipeline,
    accum: wgpu::ComputePipeline,
    bloom_first: wgpu::ComputePipeline,
    bloom_down: wgpu::ComputePipeline,
    sprite_pipe: wgpu::RenderPipeline,
    crown_pipe: wgpu::RenderPipeline,
    comp_pipe: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,

    pub accum_frames: u32,
    /// Frames to accumulate before the view counts as converged and stops tracing.
    pub accum_limit: u32,
    pub caustic_frames: u32,
    frame_counter: u32,
}

fn b(x: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    x.as_entire_binding()
}
fn tv(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}
fn sm(v: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(v)
}

fn halton(i: u32, b: u32) -> f32 {
    let (mut f, mut r, mut i) = (1.0f32, 0.0f32, i);
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

impl Renderer {
    pub fn new(gpu: &Gpu, env: Env, ground: Ground) -> Self {
        use wgpu::TextureFormat as TF;
        use B::*;
        let cs = wgpu::ShaderStages::COMPUTE;
        let samp = |addr: wgpu::AddressMode, aniso: u16| {
            gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: addr,
                address_mode_v: if addr == wgpu::AddressMode::Repeat { addr } else { wgpu::AddressMode::ClampToEdge },
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: aniso,
                ..Default::default()
            })
        };
        let lin_samp = samp(wgpu::AddressMode::ClampToEdge, 1);
        let env_samp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let ground_samp = samp(wgpu::AddressMode::Repeat, 16);
        let uni = |label: &str, size: u64| gpu.buffer(label, size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let frame_buf = uni("frame", std::mem::size_of::<FrameU>() as u64);
        let post_buf = uni("post", std::mem::size_of::<PostU>() as u64);
        let crown_buf = uni("crown", std::mem::size_of::<CrownU>() as u64);

        let rstate = gpu.storage("render state", 128);
        let wgrid = gpu.storage("wgrid", 112);
        let mgrid = gpu.storage("mgrid", 32);
        let sgrid = gpu.storage("sgrid", 32);
        let cmap = gpu.storage("cmap", 64);
        let indirect = gpu.buffer(
            "render indirect",
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        );
        let dens_acc = gpu.storage("dens acc", POOL * 512 * 4);
        let t3 = |label: &str, n: u32, format| {
            texture(
                gpu,
                label,
                (n, n, n),
                wgpu::TextureDimension::D3,
                format,
                1,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default())
        };
        let dens_view = t3("density", 288, TF::R32Float);
        let brick_range = gpu.storage("brick range", POOL * 8);
        let slot_brick = gpu.storage("slot brick", POOL * 4);
        let vindex = gpu.storage("brick index", 16);
        let mem_norm = gpu.storage("mem normals", 64 * 16);
        let mist_acc = gpu.storage("mist acc", 128 * 128 * 128 * 4);
        let mist_blur = gpu.storage("mist blur", 128 * 128 * 128 * 4);
        // Extinction and optical depth towards the sun.
        let mist_view = t3("mist", 128, TF::Rgba16Float);
        let caustic_acc = gpu.storage("caustic acc", 256 * 256 * 3 * 4);
        let sm_acc = gpu.storage("smooth acc", 128 * 128 * 128 * 10 * 4);
        let spos = gpu.storage("smoothed positions", 16);
        let aniso = gpu.storage("kernel shapes", 32);
        let caustic_view = texture(
            gpu,
            "caustics",
            (256, 256, 1),
            wgpu::TextureDimension::D2,
            TF::Rgba16Float,
            1,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());

        let prep_bgl = bgl(
            gpu,
            "prep",
            &[Uniform, Ro, Rw, Rw, Rw, Img3(TF::R32Float), Rw, Rw, Ro, Ro, Ro, Ro, Ro, Rw, Rw, Rw, Ro, Rw, Img3(TF::Rgba16Float), Rw, Rw, Rw, Rw, Rw, Rw, Rw],
            cs,
        );
        let scene_bgl = bgl(
            gpu,
            "scene",
            &[
                Uniform, Tex3, Samp, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Accel, Tex2, Samp, Tex2, Tex2, Samp, Ro, Tex2, Ro, Tex3, Ro,
                Img2(TF::Rgba16Float), Img2(TF::R32Float), Rw, Ro,
            ],
            cs,
        );
        let caustic_bgl = bgl(gpu, "caustic", &[Uniform, Rw, Img2(TF::Rgba16Float)], cs);
        let accum_bgl = bgl(gpu, "accum", &[Uniform, Tex2, Rw], cs);
        let bloom_bgl = bgl(gpu, "bloom", &[Uniform, Ro, Tex2, Samp, Img2(TF::Rgba16Float)], cs);
        let fs = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let comp_bgl = bgl(gpu, "composite", &[Uniform, Ro, Samp, Tex2, Tex2, Tex2, Tex2, Tex2, Tex2], fs);
        let sprite_bgl = bgl(gpu, "sprites", &[Uniform, Ro, Ro, Tex2, Tex2, Samp, Tex3, Samp, Ro, Ro, Ro], fs);
        let crown_bgl = bgl(gpu, "crown", &[Uniform, Uniform, Tex2, Tex2, Samp], fs);

        let common = include_str!("../shaders/common.wgsl");
        let defs = include_str!("../shaders/defs.wgsl");
        let prep_mod = gpu.shader("prep", &[common, defs, include_str!("../shaders/prep.wgsl")]);
        let water = include_str!("../shaders/water.wgsl");
        let scene_mod = gpu.shader("scene", &["enable wgpu_ray_query;\n", common, defs, water, include_str!("../shaders/scene.wgsl")]);
        let post_mod = gpu.shader("post", &[common, defs, include_str!("../shaders/post.wgsl")]);
        let accum_mod = gpu.shader("accum", &[include_str!("../shaders/accum.wgsl")]);
        let bloom_mod = gpu.shader("bloom", &[include_str!("../shaders/bloom.wgsl")]);
        let sprite_mod = gpu.shader("sprites", &[common, defs, water, include_str!("../shaders/sprites.wgsl")]);
        let crown_mod = gpu.shader("crown", &[common, defs, include_str!("../shaders/lip.wgsl"), include_str!("../shaders/crown.wgsl")]);

        let ind_bgl = bgl(gpu, "prep indirect", &[Rw], cs);
        let prep_ind = bind(gpu, &ind_bgl, &[indirect.as_entire_binding()]);
        let pl = gpu.layout(&[&prep_bgl]);
        let pl_setup = gpu.layout(&[&prep_bgl, &ind_bgl]);
        let prep = [
            "bounds_water", "mem_normals", "bounds_mist", "setup", "splat_smooth", "smooth_positions", "mark_bricks", "alloc_bricks", "splat_water", "resolve_water", "mem_indices", "splat_mist",
            "resolve_mist", "mist_shadow",
        ]
        .iter()
        .map(|&e| (e, gpu.compute(if e == "setup" { &pl_setup } else { &pl }, &prep_mod, e)))
        .collect();
        let sl = gpu.layout(&[&scene_bgl]);
        let trace = gpu.compute(&sl, &scene_mod, "trace");
        let photons = gpu.compute(&sl, &scene_mod, "photons");
        let caustic = gpu.compute(&gpu.layout(&[&caustic_bgl]), &post_mod, "caustic_resolve");
        let accum = gpu.compute(&gpu.layout(&[&accum_bgl]), &accum_mod, "accumulate");
        let bl = gpu.layout(&[&bloom_bgl]);
        let bloom_first = gpu.compute(&bl, &bloom_mod, "down_first");
        let bloom_down = gpu.compute(&bl, &bloom_mod, "down");

        let blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let overlay = |label: &str, bgl: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule| {
            gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&gpu.layout(&[bgl])),
                vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TF::Rgba16Float,
                        blend: Some(wgpu::BlendState { color: blend, alpha: blend }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sprite_pipe = overlay("sprites", &sprite_bgl, &sprite_mod);
        let crown_pipe = overlay("crown", &crown_bgl, &crown_mod);

        Self {
            env,
            ground,
            lin_samp,
            env_samp,
            ground_samp,
            frame_buf,
            post_buf,
            crown_buf,
            rstate,
            wgrid,
            mgrid,
            sgrid,
            cmap,
            indirect,
            prep_ind,
            dens_acc,
            dens_view,
            brick_range,
            slot_brick,
            vindex,
            n_vbricks: 0,
            mem_norm,
            mem_index: gpu.storage("membrane indices", 16),
            mem_accel: None,
            mist_acc,
            mist_blur,
            mist_view,
            caustic_acc,
            caustic_view,
            sm_acc,
            spos,
            aniso,
            prep_key: None,
            size: None,
            groups: None,
            prep_bgl,
            scene_bgl,
            caustic_bgl,
            accum_bgl,
            bloom_bgl,
            comp_bgl,
            sprite_bgl,
            crown_bgl,
            prep,
            trace,
            photons,
            caustic,
            accum,
            bloom_first,
            bloom_down,
            sprite_pipe,
            crown_pipe,
            comp_pipe: None,
            accum_frames: 0,
            accum_limit: MAX_ACCUM,
            caustic_frames: 0,
            frame_counter: 0,
        }
    }

    /// Call after the simulation was rebuilt; bind groups reference its buffers.
    pub fn attach(&mut self, gpu: &Gpu, cache: &Cache) {
        let sim = &cache.sim;
        self.mem_norm = gpu.storage("mem normals", sim.n_verts as u64 * 16);
        self.mem_index = gpu.buffer(
            "membrane indices",
            sim.n_tris as u64 * 12,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        );
        self.mem_accel = Some(MemAccel::new(gpu, sim.n_verts, sim.n_tris));
        self.spos = gpu.storage("smoothed positions", sim.n_particles as u64 * 16);
        self.aniso = gpu.storage("kernel shapes", sim.n_particles as u64 * 32);
        // The water density grid covers the whole MPM domain at a fixed cell size.
        let spacing = sim.scene.spacing as f32;
        let cell = 0.75 * spacing;
        let ext = Vec3::from(sim.dims.map(|d| d as f32)) * sim.dx as f32;
        let nb = (ext / (8.0 * cell)).ceil().as_uvec3();
        self.n_vbricks = nb.x * nb.y * nb.z;
        self.vindex = gpu.storage("brick index", self.n_vbricks as u64 * 4);
        let head: [[f32; 4]; 2] = [sim.grid_origin.extend(0.0).into(), [cell, cell, cell, KERNEL * spacing]];
        gpu.queue.write_buffer(&self.wgrid, 0, bytemuck::cast_slice(&head));
        gpu.queue.write_buffer(&self.wgrid, 32, bytemuck::cast_slice(&[nb.x, nb.y, nb.z, 0]));
        self.groups = None;
        self.prep_key = None;
        self.reset_accum();
    }

    pub fn reset_accum(&mut self) {
        self.accum_frames = 0;
        self.caustic_frames = 0;
    }

    pub fn reset_view_accum(&mut self) {
        self.accum_frames = 0;
    }

    pub fn converged(&self) -> bool {
        self.accum_frames >= self.accum_limit
    }

    fn ensure_size(&mut self, gpu: &Gpu, w: u32, h: u32) {
        if self.size.as_ref().is_some_and(|s| s.w == w && s.h == h) {
            return;
        }
        use wgpu::TextureFormat as TF;
        let tex2 = |label: &str, sz: (u32, u32), f: TF, extra: wgpu::TextureUsages| {
            texture(
                gpu,
                label,
                (sz.0, sz.1, 1),
                wgpu::TextureDimension::D2,
                f,
                1,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | extra,
            )
            .create_view(&Default::default())
        };
        let frame_view = tex2("frame", (w, h), TF::Rgba16Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth_view = tex2("depth", (w, h), TF::R32Float, wgpu::TextureUsages::empty());
        let accum = gpu.storage("accum", w as u64 * h as u64 * 16);
        let mut bloom = Vec::new();
        for k in 0..BLOOM_LEVELS {
            let sz = ((w >> (k + 1)).max(1), (h >> (k + 1)).max(1));
            bloom.push((tex2("bloom", sz, TF::Rgba16Float, wgpu::TextureUsages::empty()), sz));
        }
        self.size = Some(SizeRes { w, h, frame_view, depth_view, accum, bloom });
        self.groups = None;
        self.accum_frames = 0;
    }

    fn ensure_groups(&mut self, gpu: &Gpu, cache: &Cache) {
        if self.groups.is_some() {
            return;
        }
        let sim = &cache.sim;
        let s = self.size.as_ref().unwrap();
        let prep = bind(
            gpu,
            &self.prep_bgl,
            &[
                b(&self.frame_buf),
                b(&cache.particles),
                b(&self.rstate),
                b(&self.wgrid),
                b(&self.dens_acc),
                tv(&self.dens_view),
                b(&self.brick_range),
                b(&self.vindex),
                b(&cache.mem_pos),
                b(&sim.tris),
                b(&cache.alive),
                b(&sim.vtri_off),
                b(&sim.vtri),
                b(&self.mem_norm),
                b(&self.mgrid),
                b(&self.mem_index),
                b(&cache.spray),
                b(&self.mist_acc),
                tv(&self.mist_view),
                b(&self.sgrid),
                b(&self.cmap),
                b(&self.sm_acc),
                b(&self.spos),
                b(&self.aniso),
                b(&self.slot_brick),
                b(&self.mist_blur),
            ],
        );
        let scene = bind(
            gpu,
            &self.scene_bgl,
            &[
                b(&self.frame_buf),
                tv(&self.dens_view),
                sm(&self.lin_samp),
                b(&self.wgrid),
                b(&self.vindex),
                b(&cache.mem_pos),
                b(&self.mem_norm),
                b(&sim.tris),
                b(&cache.alive),
                b(&self.mgrid),
                wgpu::BindingResource::AccelerationStructure(&self.mem_accel.as_ref().unwrap().tlas),
                tv(&self.env.view),
                sm(&self.env_samp),
                tv(&self.ground.albedo),
                tv(&self.ground.detail),
                sm(&self.ground_samp),
                b(&cache.wet),
                tv(&self.caustic_view),
                b(&self.cmap),
                tv(&self.mist_view),
                b(&self.sgrid),
                tv(&s.frame_view),
                tv(&s.depth_view),
                b(&self.caustic_acc),
                b(&self.brick_range),
            ],
        );
        let caustic = bind(gpu, &self.caustic_bgl, &[b(&self.frame_buf), b(&self.caustic_acc), tv(&self.caustic_view)]);
        let accum = bind(gpu, &self.accum_bgl, &[b(&self.post_buf), tv(&s.frame_view), b(&s.accum)]);
        let mut bloom = Vec::new();
        for k in 0..BLOOM_LEVELS {
            let src = if k == 0 { &s.bloom[BLOOM_LEVELS - 1].0 } else { &s.bloom[k - 1].0 };
            bloom.push(bind(
                gpu,
                &self.bloom_bgl,
                &[b(&self.post_buf), b(&s.accum), tv(src), sm(&self.lin_samp), tv(&s.bloom[k].0)],
            ));
        }
        let comp = bind(
            gpu,
            &self.comp_bgl,
            &[
                b(&self.post_buf),
                b(&s.accum),
                sm(&self.lin_samp),
                tv(&s.bloom[0].0),
                tv(&s.bloom[1].0),
                tv(&s.bloom[2].0),
                tv(&s.bloom[3].0),
                tv(&s.bloom[4].0),
                tv(&s.bloom[5].0),
            ],
        );
        let sprite = bind(
            gpu,
            &self.sprite_bgl,
            &[
                b(&self.frame_buf),
                b(&cache.spray),
                b(&cache.particles),
                tv(&s.depth_view),
                tv(&self.env.view),
                sm(&self.env_samp),
                tv(&self.dens_view),
                sm(&self.lin_samp),
                b(&self.wgrid),
                b(&self.spos),
                b(&self.vindex),
            ],
        );
        let crown = bind(gpu, &self.crown_bgl, &[b(&self.frame_buf), b(&self.crown_buf), tv(&s.depth_view), tv(&self.env.view), sm(&self.env_samp)]);
        self.groups = Some(Groups { prep, scene, caustic, accum, bloom, comp, sprite, crown });
    }

    fn comp_pipeline(&mut self, gpu: &Gpu, format: wgpu::TextureFormat) {
        if self.comp_pipe.as_ref().is_some_and(|(f, _)| *f == format) {
            return;
        }
        let m = gpu.shader("composite", &[include_str!("../shaders/composite.wgsl")]);
        let p = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
            layout: Some(&gpu.layout(&[&self.comp_bgl])),
            vertex: wgpu::VertexState { module: &m, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &m,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.comp_pipe = Some((format, p));
    }

    pub fn base_exposure(&self) -> f32 {
        let sun = self.env.sun_irr;
        let e_sun = 0.2126 * sun.x + 0.7152 * sun.y + 0.0722 * sun.z;
        let lum = 0.3 / std::f32::consts::PI * (e_sun * self.env.sun_dir_env.y.max(0.2) + self.env.sky_irr_up);
        0.18 / lum.max(1e-6)
    }

    pub fn sun_dir_world(&self, look: &LookParams) -> Vec3 {
        let a = look.env_rotation_deg.to_radians();
        let s = self.env.sun_dir_env;
        Vec3::new(a.cos() * s.x + a.sin() * s.z, s.y, -a.sin() * s.x + a.cos() * s.z)
    }

    /// Renders one progressive sample (unless converged) and composites it into `target`.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        gpu: &Gpu,
        cache: &Cache,
        cam: &Camera,
        look: &LookParams,
        render_size: (u32, u32),
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        rect: [f32; 4],
        clear: wgpu::Color,
    ) {
        let (w, h) = (render_size.0.max(8), render_size.1.max(8));
        self.ensure_size(gpu, w, h);
        self.ensure_groups(gpu, cache);
        self.comp_pipeline(gpu, target_format);
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let trace_now = !self.converged();
        let sim = &cache.sim;
        let t = cache.shown;

        let fwd = (cam.target - cam.position()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = 12.0 / look.focal_mm;
        let pixel_angle = 2.0 * tan_half / h as f32;
        let env_texel = 2.0 * std::f32::consts::PI / 4096.0;
        let lod_bg = (pixel_angle / env_texel).log2().max(0.0);
        // Depth of field builds up while accumulating: every frame looks through a different
        // point of the aperture at the same focus plane. A single frame is a pinhole.
        let (jit, lens) = if self.accum_frames == 0 {
            ([0.0, 0.0], Vec3::ZERO)
        } else {
            let n = self.accum_frames;
            let aperture = if look.dof { look.focal_mm * 1e-3 / look.f_stop } else { 0.0 };
            let (r, a) = (halton(n, 5).sqrt() * 0.5 * aperture, halton(n, 7) * std::f32::consts::TAU);
            ([halton(n, 2) - 0.5, halton(n, 3) - 0.5], right * (r * a.cos()) + up * (r * a.sin()))
        };
        let sun_dir = self.sun_dir_world(look);
        let sun_col = self.env.sun_irr * look.sun_intensity;
        let mut bullets = [BulletU::zeroed(); MAX_SHOTS];
        for (bu, s) in bullets.iter_mut().zip(&sim.shots) {
            let (f, b) = (&s.flight, &s.flight.body);
            let (tip, axis, roll) = f.pose_at(t);
            *bu = BulletU {
                pos: tip.as_vec3().extend(b.radius as f32).into(),
                dir: axis.as_vec3().extend(b.length as f32).into(),
                look: [b.nose as u32 as f32, roll.rem_euclid(std::f64::consts::TAU) as f32, f.twist as f32, b.metal as u32 as f32],
            };
        }
        let c = Vec3::from(look.balloon_color).clamp(Vec3::splat(0.02), Vec3::splat(0.98));
        let t_int = sim.scene.thickness0 as f32 * 1e3 / (sim.scene.stretch * sim.scene.stretch) as f32;
        let sigma_a = -Vec3::new(c.x.ln(), c.y.ln(), c.z.ln()) / t_int;
        let albedo = c.lerp(Vec3::ONE, 0.12);
        let spacing = sim.scene.spacing as f32;
        let mut sh = [[0f32; 4]; 9];
        sh.copy_from_slice(&self.env.sh);
        let caustics = look.caustics;
        let fu = FrameU {
            cam_pos: (cam.position() + lens).extend(tan_half).into(),
            cam_fwd: fwd.extend(w as f32 / h as f32).into(),
            cam_right: right.extend(0.0).into(),
            cam_up: up.extend(0.0).into(),
            res: [w as f32, h as f32, self.accum_frames as f32, pixel_angle],
            jitter: [jit[0], jit[1], f32::from_bits(self.frame_counter.wrapping_mul(2654435761)), lod_bg],
            sun_dir: sun_dir.extend(0.00465).into(),
            sun_col: sun_col.extend(look.env_rotation_deg.to_radians()).into(),
            sky: [look.sky_intensity, 1.0, look.bloom, 2.5],
            sh,
            knot: sim.scene.knot.extend((sim.scene.profile.neck_r as f32 * 1.35).max(0.006)).into(),
            rubber: sigma_a.extend(sim.scene.thickness0 as f32 * 1e3).into(),
            // Pigmented latex is turbid: a stretched skin is milky, a relaxed scrap opaque.
            rubber2: albedo.extend(12.0).into(),
            water: [0.5, spacing, KERNEL * spacing, 0.75 * spacing],
            wet: [WET_HALF, WET_RES as f32, 4.0, 0.0006],
            flags: [caustics as u32, look.max_bounces, look.mist as u32, sim.n_tris],
            counts: [sim.n_particles, cache.shown_spray, sim.n_verts, PHOTON_N],
            // Lattice sum of the (1 - d^2/R^2)^3 kernel normalises interior density to 1.
            time: [t as f32, look.shutter_us * 1e-6, (self.caustic_frames + 1) as f32, 0.9574 * KERNEL * KERNEL * KERNEL],
            lens: (-lens / cam.dist).extend(0.0).into(),
            bullets,
        };
        gpu.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&fu));
        let mut cu = CrownU::zeroed();
        let spray_on = (sim.params.spray_amount > 0.0) as u32 as f32;
        for (m, lip) in sim.shots.iter().flat_map(|s| &s.lips).enumerate() {
            if let Some(l) = lip {
                cu.mouth[m] = l.center.extend(l.k as f32).into();
                cu.axis[m] = l.axis.extend(l.t0 as f32).into();
                cu.lip[m] = [l.rb as f32, l.tau_end as f32, spray_on, 0.0];
            }
        }
        cu.misc = [t as f32, P_ATM as f32, TAU_VENT as f32, sim.scene.radius_max as f32];
        gpu.queue.write_buffer(&self.crown_buf, 0, bytemuck::bytes_of(&cu));
        let exposure = self.base_exposure() * 2f32.powf(look.exposure_ev);
        let pu = PostU {
            rect,
            res: [w as f32, h as f32, exposure, look.bloom],
            misc: [(self.frame_counter % 1024) as f32 * 0.37, 0.22, self.accum_frames as f32, 0.0],
            _pad: [0.0; 4],
        };
        gpu.queue.write_buffer(&self.post_buf, 0, bytemuck::bytes_of(&pu));
        let prep_key = (cache.shown_id, sun_dir.to_array().map(f32::to_bits));
        let prep_now = trace_now && self.prep_key != Some(prep_key);
        let mut rs = [0i32; 32];
        for k in 0..3 {
            for base in [0, 8, 16] {
                rs[base + k] = i32::MAX / 2;
                rs[base + 4 + k] = i32::MIN / 2;
            }
        }
        if prep_now {
            gpu.queue.write_buffer(&self.rstate, 0, bytemuck::cast_slice(&rs));
            self.prep_key = Some(prep_key);
        }

        let g = self.groups.as_ref().unwrap();
        let s = self.size.as_ref().unwrap();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        if prep_now {
            enc.clear_buffer(&self.vindex, 0, None);
            enc.clear_buffer(&self.mist_acc, 0, None);
            enc.clear_buffer(&self.sm_acc, 0, None);
        }
        if trace_now {
            if self.caustic_frames == 0 {
                enc.clear_buffer(&self.caustic_acc, 0, None);
            }
            if prep_now {
                gpu.mark(&mut enc, "prep");
                let mut cp = enc.begin_compute_pass(&Default::default());
                {
                    cp.set_bind_group(0, &g.prep, &[]);
                    let pp = |name: &str| &self.prep.iter().find(|(n, _)| *n == name).unwrap().1;
                    let (pgx, pgy) = groups_1d(sim.n_particles);
                    let (sgx, sgy) = groups_1d(cache.shown_spray);
                    cp.set_pipeline(pp("bounds_water"));
                    cp.dispatch_workgroups(pgx, pgy, 1);
                    cp.set_pipeline(pp("mem_normals"));
                    cp.dispatch_workgroups(sim.n_verts.div_ceil(256), 1, 1);
                    cp.set_pipeline(pp("bounds_mist"));
                    cp.dispatch_workgroups(sgx, sgy, 1);
                    cp.set_bind_group(1, &self.prep_ind, &[]);
                    cp.set_pipeline(pp("setup"));
                    cp.dispatch_workgroups(1, 1, 1);
                    drop(cp);
                    gpu.mark(&mut enc, "prep splat");
                    cp = enc.begin_compute_pass(&Default::default());
                    cp.set_bind_group(0, &g.prep, &[]);
                    cp.set_pipeline(pp("splat_smooth"));
                    cp.dispatch_workgroups(pgx, pgy, 1);
                    drop(cp);
                    gpu.mark(&mut enc, "prep smooth");
                    cp = enc.begin_compute_pass(&Default::default());
                    cp.set_bind_group(0, &g.prep, &[]);
                    cp.set_pipeline(pp("smooth_positions"));
                    cp.dispatch_workgroups(pgx, pgy, 1);
                    drop(cp);
                    gpu.mark(&mut enc, "prep water");
                    cp = enc.begin_compute_pass(&Default::default());
                    cp.set_bind_group(0, &g.prep, &[]);
                    cp.set_pipeline(pp("mark_bricks"));
                    cp.dispatch_workgroups(pgx, pgy, 1);
                    let (vgx, vgy) = groups_1d(self.n_vbricks);
                    cp.set_pipeline(pp("alloc_bricks"));
                    cp.dispatch_workgroups(vgx, vgy, 1);
                    cp.set_pipeline(pp("splat_water"));
                    cp.dispatch_workgroups(pgx, pgy, 1);
                    cp.set_pipeline(pp("resolve_water"));
                    cp.dispatch_workgroups(POOL as u32, 1, 1);
                    drop(cp);
                    gpu.mark(&mut enc, "prep mem mist");
                    cp = enc.begin_compute_pass(&Default::default());
                    cp.set_bind_group(0, &g.prep, &[]);
                    cp.set_bind_group(1, &self.prep_ind, &[]);
                    cp.set_pipeline(pp("mem_indices"));
                    cp.dispatch_workgroups(sim.n_tris.div_ceil(256), 1, 1);
                    cp.set_pipeline(pp("splat_mist"));
                    cp.dispatch_workgroups(sgx, sgy, 1);
                    cp.set_pipeline(pp("resolve_mist"));
                    cp.dispatch_workgroups_indirect(&self.indirect, 12);
                    cp.set_pipeline(pp("mist_shadow"));
                    cp.dispatch_workgroups_indirect(&self.indirect, 12);
                }
            }
            if prep_now {
                gpu.mark(&mut enc, "accel");
                self.mem_accel.as_ref().unwrap().build(&mut enc, &cache.mem_pos, &self.mem_index);
            }
            if caustics {
                gpu.mark(&mut enc, "caustics");
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &g.scene, &[]);
                cp.set_pipeline(&self.photons);
                cp.dispatch_workgroups(PHOTON_N / 8, PHOTON_N / 8, 1);
                cp.set_bind_group(0, &g.caustic, &[]);
                cp.set_pipeline(&self.caustic);
                cp.dispatch_workgroups(32, 32, 1);
            }
            {
                gpu.mark(&mut enc, "trace");
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &g.scene, &[]);
                cp.set_pipeline(&self.trace);
                cp.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
            }
            gpu.mark(&mut enc, "sprites");
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sprites"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &s.frame_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(&self.crown_pipe);
                rp.set_bind_group(0, &g.crown, &[]);
                rp.draw(0..2 * sim.shots.len() as u32 * CROWN_VERTS, 0..1);
                rp.set_pipeline(&self.sprite_pipe);
                rp.set_bind_group(0, &g.sprite, &[]);
                rp.draw(0..6, 0..cache.shown_spray + sim.n_particles);
            }
            gpu.mark(&mut enc, "post");
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &g.accum, &[]);
                cp.set_pipeline(&self.accum);
                cp.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
                for k in 0..BLOOM_LEVELS {
                    cp.set_bind_group(0, &g.bloom[k], &[]);
                    cp.set_pipeline(if k == 0 { &self.bloom_first } else { &self.bloom_down });
                    let (bw, bh) = s.bloom[k].1;
                    cp.dispatch_workgroups(bw.div_ceil(8), bh.div_ceil(8), 1);
                }
            }
            self.accum_frames += 1;
            if caustics {
                self.caustic_frames += 1;
            }
        }
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_viewport(rect[0], rect[1], rect[2], rect[3], 0.0, 1.0);
            rp.set_pipeline(&self.comp_pipe.as_ref().unwrap().1);
            rp.set_bind_group(0, &g.comp, &[]);
            rp.draw(0..3, 0..1);
        }
        gpu.mark(&mut enc, "-");
        gpu.queue.submit([enc.finish()]);
    }
}
