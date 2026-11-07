use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Ceiling on this process's GPU memory. Everything is sized up front (the keyframe cache is
/// one fixed pool), so crossing it means a bug. Stopping the app then is far better than
/// filling VRAM, which makes the kernel refuse the compositor's submissions too.
const GPU_MEMORY_CEILING: u64 = 5 << 30;

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub timer: RefCell<GpuTimer>,
}

impl Gpu {
    pub fn new(surface: Option<&wgpu::Surface<'_>>, instance: &wgpu::Instance) -> anyhow::Result<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
        }))?;
        // Ray queries trace the membrane on the ray tracing hardware.
        let features = wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        anyhow::ensure!(adapter.features().contains(features), "GPU lacks {:?}", features - adapter.features());
        let ts = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let ts = if adapter.features().contains(ts) { ts } else { wgpu::Features::empty() };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("natray"),
            required_features: features | ts,
            required_limits: adapter.limits(),
            // SAFETY: only ray queries are used from the experimental set.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))?;
        let timer = RefCell::new(GpuTimer::new(&device, &queue, !ts.is_empty()));
        Ok(Self { adapter, device, queue, timer })
    }

    /// Timestamps the point between two passes. `label` names the span up to the next mark,
    /// "-" ends a span without starting one.
    pub fn mark(&self, enc: &mut wgpu::CommandEncoder, label: &'static str) {
        self.timer.borrow_mut().mark(enc, label);
    }

    /// GPU milliseconds spent under `label` in the last frame that was read back.
    pub fn span_ms(&self, label: &str) -> Option<f64> {
        self.timer.borrow().spans.iter().find(|(l, _)| *l == label).map(|s| s.1)
    }

    /// GPU memory this process holds, when the backend reports it.
    pub fn memory_used(&self) -> Option<u64> {
        self.device.generate_allocator_report().map(|r| r.total_reserved_bytes)
    }

    /// Every allocation passes through here first.
    pub fn reserve(&self, label: &str, bytes: u64) {
        let Some(used) = self.memory_used() else { return };
        assert!(
            used + bytes <= GPU_MEMORY_CEILING,
            "refusing to allocate {label} ({} MB): {} MB of GPU memory already in use, ceiling {} MB",
            bytes >> 20,
            used >> 20,
            GPU_MEMORY_CEILING >> 20
        );
    }

    pub fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.reserve(label, size);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.buffer(
            label,
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub fn storage_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let mut padded;
        let contents = if data.len() < 16 {
            padded = data.to_vec();
            padded.resize(16, 0);
            &padded[..]
        } else {
            data
        };
        self.reserve(label, contents.len() as u64);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn shader(&self, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
        let src: String = parts.concat();
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
        })
    }

    pub fn compute(&self, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    pub fn layout(&self, bgls: &[&wgpu::BindGroupLayout]) -> wgpu::PipelineLayout {
        let v: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(|b| Some(*b)).collect();
        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &v,
            immediate_size: 0,
        })
    }
}

/// Compact description of a bind group layout entry.
#[derive(Clone, Copy)]
pub enum B {
    Uniform,
    UniformDyn,
    Rw,
    Ro,
    Tex2,
    Tex3,
    Samp,
    Img2(wgpu::TextureFormat),
    Img3(wgpu::TextureFormat),
    Accel,
}

pub fn bgl(gpu: &Gpu, label: &str, entries: &[B], vis: wgpu::ShaderStages) -> wgpu::BindGroupLayout {
    use wgpu::{BindingType as T, BufferBindingType as BB};
    let e: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let ty = match *b {
                B::Uniform => T::Buffer { ty: BB::Uniform, has_dynamic_offset: false, min_binding_size: None },
                B::UniformDyn => T::Buffer { ty: BB::Uniform, has_dynamic_offset: true, min_binding_size: None },
                B::Rw => T::Buffer { ty: BB::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                B::Ro => T::Buffer { ty: BB::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                B::Tex2 => T::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                B::Tex3 => T::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                B::Samp => T::Sampler(wgpu::SamplerBindingType::Filtering),
                B::Img2(f) => T::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: f,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                B::Img3(f) => T::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: f,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                B::Accel => T::AccelerationStructure { vertex_return: false },
            };
            // Read-write storage is not allowed in vertex shaders, so only expose it where it can be used.
            let visibility = match *b {
                B::Rw | B::Img2(_) | B::Img3(_) => vis & !wgpu::ShaderStages::VERTEX,
                _ => vis,
            };
            wgpu::BindGroupLayoutEntry { binding: i as u32, visibility, ty, count: None }
        })
        .collect();
    gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &e })
}

pub fn bind(gpu: &Gpu, layout: &wgpu::BindGroupLayout, res: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let e: Vec<wgpu::BindGroupEntry> = res
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() })
        .collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &e })
}

/// Splits a large 1D dispatch into x/y so it stays under the per-dimension limit.
/// Shaders reconstruct the index with `gid.x + gid.y * num_workgroups.x * 256`.
pub fn groups_1d(n: u32) -> (u32, u32) {
    let g = n.div_ceil(256).max(1);
    if g <= 65535 {
        (g, 1)
    } else {
        (65535, g.div_ceil(65535))
    }
}

pub fn texture(
    gpu: &Gpu,
    label: &str,
    size: (u32, u32, u32),
    dim: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    mips: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    let texel = format.block_copy_size(None).unwrap_or(16) as u64;
    // A full mip chain adds at most as much again.
    let mip_factor = if mips > 1 { 2 } else { 1 };
    gpu.reserve(label, size.0 as u64 * size.1 as u64 * size.2 as u64 * texel * mip_factor);
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: dim,
        format,
        usage,
        view_formats: &[],
    })
}

pub fn read_buffer(gpu: &Gpu, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
    read_buffer_at(gpu, buf, 0, size)
}

pub fn read_buffer_at(gpu: &Gpu, buf: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
    let staging = gpu.buffer("readback", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, offset, &staging, 0, size);
    gpu.queue.submit([enc.finish()]);
    let slice = staging.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    out
}

const TIMER_SLOTS: u32 = 64;

/// GPU timestamps between named points of a frame. Results are read back a frame or two
/// later without stalling. Stays empty on adapters without timestamp support.
pub struct GpuTimer {
    set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    marks: Vec<&'static str>,
    resolved: u32,
    in_flight: bool,
    mapped: Arc<AtomicBool>,
    period_ns: f32,
    /// Milliseconds per label, summed over the frame.
    pub spans: Vec<(&'static str, f64)>,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> Self {
        let set = enabled.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timer"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMER_SLOTS,
            })
        });
        let buf = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timer"),
                size: TIMER_SLOTS as u64 * 8,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            set,
            resolve: buf(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buf(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            marks: Vec::new(),
            resolved: 0,
            in_flight: false,
            mapped: Arc::new(AtomicBool::new(false)),
            period_ns: queue.get_timestamp_period(),
            spans: Vec::new(),
        }
    }

    fn mark(&mut self, enc: &mut wgpu::CommandEncoder, label: &'static str) {
        let Some(set) = &self.set else { return };
        if self.in_flight || self.resolved > 0 || self.marks.len() as u32 >= TIMER_SLOTS {
            return;
        }
        enc.write_timestamp(set, self.marks.len() as u32);
        self.marks.push(label);
    }

    /// Records the readback into the frame's last encoder. Returns whether this frame will
    /// be timed.
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) -> bool {
        let Some(set) = &self.set else { return false };
        let n = self.marks.len() as u32;
        if self.in_flight || n < 2 {
            return false;
        }
        enc.resolve_query_set(set, 0..n, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
        self.resolved = n;
        true
    }

    /// Call once the encoder given to `resolve` has been submitted.
    pub fn submitted(&mut self) {
        if self.resolved == 0 || self.in_flight {
            return;
        }
        self.in_flight = true;
        let flag = self.mapped.clone();
        self.read.slice(..self.resolved as u64 * 8).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
    }

    /// Picks up a finished readback and starts a new frame of marks. Returns whether new
    /// spans arrived.
    pub fn begin_frame(&mut self, device: &wgpu::Device) -> bool {
        let mut fresh = false;
        if self.in_flight {
            device.poll(wgpu::PollType::Poll).ok();
            if !self.mapped.swap(false, Ordering::Acquire) {
                return false;
            }
            let n = self.resolved as usize;
            let ticks: Vec<u64> = bytemuck::cast_slice(&self.read.slice(..n as u64 * 8).get_mapped_range()).to_vec();
            self.read.unmap();
            self.spans.clear();
            for i in 0..n - 1 {
                let label = self.marks[i];
                if label == "-" {
                    continue;
                }
                let ms = ticks[i + 1].saturating_sub(ticks[i]) as f64 * self.period_ns as f64 * 1e-6;
                match self.spans.iter_mut().find(|(l, _)| *l == label) {
                    Some(s) => s.1 += ms,
                    None => self.spans.push((label, ms)),
                }
            }
            self.in_flight = false;
            fresh = true;
        }
        self.marks.clear();
        self.resolved = 0;
        fresh
    }
}
