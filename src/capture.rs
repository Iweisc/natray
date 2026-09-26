//! Headless rendering of stills: `natray --capture DIR [--preset NAME] --shot t=..,az=..,el=..,dist=.. [...]`.

use crate::env::{load_env, load_ground};
use crate::gpu::{texture, Gpu};
use crate::params::{set_kv, LookParams, SimParams};
use crate::cache::Cache;
use crate::presets::{self, View};
use crate::render::{Camera, Renderer};
use glam::Vec3;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Shot {
    pub t: f64,
    pub cam: Camera,
    pub focal: Option<f32>,
    pub name: String,
}

fn parse_shot(s: &str, default_target: Vec3, view: View) -> anyhow::Result<Shot> {
    let mut shot = Shot {
        t: 0.0,
        cam: Camera { target: default_target, yaw: view.yaw, pitch: view.pitch, dist: view.dist },
        focal: None,
        name: String::new(),
    };
    for kv in s.split(',') {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("bad shot field '{kv}'"))?;
        let f: f32 = if k == "name" { 0.0 } else { v.parse()? };
        match k {
            "t" => shot.t = v.parse()?,
            "az" => shot.cam.yaw = f,
            "el" => shot.cam.pitch = f,
            "dist" => shot.cam.dist = f,
            "tx" => shot.cam.target.x = f,
            "ty" => shot.cam.target.y = f,
            "tz" => shot.cam.target.z = f,
            "focal" => shot.focal = Some(f),
            "name" => shot.name = v.to_string(),
            _ => anyhow::bail!("unknown shot field '{k}'"),
        }
    }
    Ok(shot)
}

pub fn assets_dir() -> PathBuf {
    std::env::var_os("NATRAY_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets")))
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let mut out = PathBuf::from("captures");
    let mut size = (1600u32, 900u32);
    let mut spp = 64u32;
    let mut bench = 0usize;
    let mut shot_specs = Vec::new();
    let mut sim_p = SimParams::default();
    let mut look = LookParams::default();
    let mut view = View { yaw: -35.0, pitch: 8.0, dist: 1.3 };
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let val = || args.get(i + 1).cloned().ok_or_else(|| anyhow::anyhow!("{a} needs a value"));
        match a {
            "--capture" => out = PathBuf::from(val()?),
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or_else(|| anyhow::anyhow!("--size WxH"))?;
                size = (w.parse()?, h.parse()?);
            }
            "--spp" => spp = val()?.parse()?,
            "--bench" => bench = val()?.parse()?,
            "--shot" => shot_specs.push(val()?),
            // Before any --set, which would otherwise be overwritten.
            "--preset" => {
                let p = presets::named(&val()?)?;
                (sim_p, look, view) = (p.sim, p.look, p.view);
            }
            "--set" => {
                let v = val()?;
                let (k, x) = v.split_once('=').ok_or_else(|| anyhow::anyhow!("--set key=value"))?;
                set_kv(&mut sim_p, &mut look, k, x)?;
            }
            _ => anyhow::bail!("unknown argument '{a}'"),
        }
        i += 2;
    }
    std::fs::create_dir_all(&out)?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let gpu = Gpu::new(None, &instance)?;
    let assets = assets_dir();
    let env = load_env(&gpu, &assets.join("env.hdr"))?;
    let ground = load_ground(&gpu, &assets)?;
    println!("sun env dir {:?} irradiance {:?} sky {:.1}", env.sun_dir_env, env.sun_irr, env.sky_irr_up);
    let mut cache = Cache::new(&gpu, &sim_p);
    let mut renderer = Renderer::new(&gpu, env, ground);
    renderer.attach(&gpu, &cache);
    let target_default = cache.sim.scene.center;
    let mut shots: Vec<Shot> = shot_specs.iter().map(|s| parse_shot(s, target_default, view)).collect::<anyhow::Result<_>>()?;
    if shots.is_empty() {
        shots.push(parse_shot("t=0.002", target_default, view)?);
    }
    shots.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

    let tex = offscreen(&gpu, size);
    let view = tex.create_view(&Default::default());
    let mut chunk = 4usize;

    for (k, shot) in shots.iter().enumerate() {
        let t0 = Instant::now();
        let mut steps = 0;
        // A benchmark plays on from the shot, so bake that far too.
        let bake_to = shot.t + bench as f64 * 0.01;
        while cache.end() < bake_to && !cache.done() {
            let n = cache.bake(&gpu, chunk);
            let ms = breathe(&gpu);
            steps += n;
            if n > 0 {
                chunk = ((chunk as f64 * CHUNK_MS / ms.max(0.1)).round() as usize).clamp(1, 64);
            }
        }
        let sim_ms = t0.elapsed().as_secs_f64() * 1e3;
        if bench > 0 {
            bench_playback(&gpu, &mut cache, &mut renderer, shot, &look, size, &view, bench);
            continue;
        }
        if std::env::var_os("NATRAY_DEBUG").is_some() {
            cache.sim.debug_stats(&gpu);
        }
        let mut l = look.clone();
        if let Some(f) = shot.focal {
            l.focal_mm = f;
        }
        cache.show(&gpu, shot.t);
        if let Some(d) = std::env::var_os("NATRAY_DUMP") {
            let n = cache.sim.n_particles as u64;
            std::fs::write(format!("{}_{}_p.bin", d.to_string_lossy(), shot.name), crate::gpu::read_buffer(&gpu, &cache.particles, n * 32))?;
            std::fs::write(format!("{}_{}_s.bin", d.to_string_lossy(), shot.name), crate::gpu::read_buffer(&gpu, &cache.spray, cache.shown_spray as u64 * 32))?;
            for s in &cache.sim.shots {
                if let Some(p) = &s.flight.path {
                    println!("DUMP origin {:?} dir {:?} a_out {} rb {} t_through {}", p.origin, p.dir, p.a_out, s.flight.body.radius, p.t_through);
                }
            }
        }
        renderer.reset_accum();
        let t1 = Instant::now();
        let mut first = String::new();
        for i in 0..spp {
            gpu.timer.borrow_mut().begin_frame(&gpu.device);
            if i == 1 {
                first = spans(&gpu);
            }
            renderer.render(&gpu, &cache, &shot.cam, &l, size, &view, OFFSCREEN_FORMAT, [0.0, 0.0, size.0 as f32, size.1 as f32], wgpu::Color::BLACK);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            gpu.timer.borrow_mut().resolve(&mut enc);
            gpu.queue.submit([enc.finish()]);
            gpu.timer.borrow_mut().submitted();
            breathe(&gpu);
        }
        gpu.timer.borrow_mut().begin_frame(&gpu.device);
        let img = read_image(&gpu, &tex, size);
        let render_ms = t1.elapsed().as_secs_f64() * 1e3;
        let name = if shot.name.is_empty() {
            format!("shot{:02}_t{:.4}_az{:.0}_el{:.0}.png", k, shot.t, shot.cam.yaw, shot.cam.pitch)
        } else {
            format!("{}.png", shot.name)
        };
        let path = out.join(name);
        save_png(img, &path)?;
        println!(
            "{}  baked to {:.5}s ({} steps, {:.0} ms wall)  render {} spp {:.0} ms wall\n  first frame: {}\n  last frame:  {}",
            path.display(),
            cache.end(),
            steps,
            sim_ms,
            spp,
            render_ms,
            first,
            spans(&gpu)
        );
    }
    println!("GPU memory {:?} MB", gpu.memory_used().map(|b| b >> 20));
    println!(
        "cache {} keys, {:.0} of {:.0} MB{}",
        cache.key_count(),
        cache.used_bytes() as f64 / 1e6,
        cache.pool_bytes() as f64 / 1e6,
        if cache.full() { ", full" } else { "" }
    );
    Ok(())
}

/// Plays `frames` frames from the shot at 1x and 100 fps, back to back like the app, and
/// prints the median GPU time of each pass. One frame stays queued behind the one being
/// waited for. An idle GPU clocks down between frames and the timings then swing by 2x.
fn bench_playback(
    gpu: &Gpu,
    cache: &mut Cache,
    renderer: &mut Renderer,
    shot: &Shot,
    look: &LookParams,
    size: (u32, u32),
    view: &wgpu::TextureView,
    frames: usize,
) {
    let mut samples: Vec<(&'static str, Vec<f64>)> = Vec::new();
    let mut prev = None;
    for f in 0..frames {
        let fresh = gpu.timer.borrow_mut().begin_frame(&gpu.device);
        if fresh {
            for &(l, ms) in &gpu.timer.borrow().spans {
                match samples.iter_mut().find(|(k, _)| *k == l) {
                    Some(v) => v.1.push(ms),
                    None => samples.push((l, vec![ms])),
                }
            }
        }
        cache.show(gpu, shot.t + f as f64 * 0.01);
        renderer.reset_accum();
        renderer.render(gpu, cache, &shot.cam, look, size, view, OFFSCREEN_FORMAT, [0.0, 0.0, size.0 as f32, size.1 as f32], wgpu::Color::BLACK);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        gpu.timer.borrow_mut().resolve(&mut enc);
        let idx = gpu.queue.submit([enc.finish()]);
        gpu.timer.borrow_mut().submitted();
        if let Some(p) = prev.replace(idx) {
            gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(p), timeout: None }).ok();
        }
    }
    let mut total = 0.0;
    let mut line = Vec::new();
    for (l, v) in &mut samples {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = v[v.len() / 2];
        total += m;
        line.push(format!("{l} {m:.2}"));
    }
    println!("t={:.3} {}x{}: {:.2} ms  ({})", shot.t, size.0, size.1, total, line.join(", "));
}

/// GPU milliseconds per bake chunk. Short chunks let the compositor in between.
const CHUNK_MS: f64 = 4.0;

/// Waits for the GPU, then stays idle about as long as it was busy, so a long capture
/// doesn't starve the desktop. Returns the busy time in ms.
fn breathe(gpu: &Gpu) -> f64 {
    let t0 = Instant::now();
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let busy = t0.elapsed();
    std::thread::sleep(busy);
    busy.as_secs_f64() * 1e3
}

fn spans(gpu: &Gpu) -> String {
    gpu.timer.borrow().spans.iter().map(|(l, ms)| format!("{l} {ms:.2}")).collect::<Vec<_>>().join(", ")
}

pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub fn offscreen(gpu: &Gpu, size: (u32, u32)) -> wgpu::Texture {
    texture(
        gpu,
        "offscreen",
        (size.0, size.1, 1),
        wgpu::TextureDimension::D2,
        OFFSCREEN_FORMAT,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    )
}

/// Reads an offscreen texture back to the CPU. Blocks until the GPU is done.
pub fn read_image(gpu: &Gpu, tex: &wgpu::Texture, size: (u32, u32)) -> image::RgbaImage {
    let row = (size.0 * 4).div_ceil(256) * 256;
    let readback = gpu.buffer("readback", (row * size.1) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.1) },
        },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mut img = image::RgbaImage::new(size.0, size.1);
    {
        let data = slice.get_mapped_range();
        for y in 0..size.1 {
            let src = &data[(y * row) as usize..][..(size.0 * 4) as usize];
            img.as_mut()[(y * size.0 * 4) as usize..][..(size.0 * 4) as usize].copy_from_slice(src);
        }
    }
    readback.unmap();
    img
}

pub fn save_png(img: image::RgbaImage, path: &Path) -> anyhow::Result<()> {
    image::DynamicImage::ImageRgba8(img).to_rgb8().save(path)?;
    Ok(())
}
