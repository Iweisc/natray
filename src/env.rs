//! Loads the HDR environment and ground textures. The sun is cut out of the HDRI and
//! rendered analytically so it can cast sharp shadows and caustics.

use crate::gpu::{texture, Gpu};
use glam::Vec3;
use std::f32::consts::PI;
use std::path::Path;

pub struct Env {
    pub view: wgpu::TextureView,
    /// Convolved SH9 irradiance coefficients (rgb) in env space.
    pub sh: [[f32; 4]; 9],
    /// Direction to the sun in env space.
    pub sun_dir_env: Vec3,
    /// Sun irradiance at normal incidence (W/m^2-ish, HDRI units).
    pub sun_irr: Vec3,
    pub sky_irr_up: f32,
}

pub struct Ground {
    pub albedo: wgpu::TextureView,
    pub detail: wgpu::TextureView,
}

fn dir_of(u: f32, v: f32) -> Vec3 {
    let phi = (u - 0.5) * 2.0 * PI;
    let th = v * PI;
    Vec3::new(th.sin() * phi.cos(), th.cos(), th.sin() * phi.sin())
}

fn sh_basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

fn downsample(src: &[[f32; 4]], w: usize, h: usize) -> (Vec<[f32; 4]>, usize, usize) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![[0f32; 4]; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut s = [0f32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = src[(y * 2 + dy).min(h - 1) * w + (x * 2 + dx).min(w - 1)];
                for k in 0..4 {
                    s[k] += 0.25 * p[k];
                }
            }
            out[y * nw + x] = s;
        }
    }
    (out, nw, nh)
}

pub fn load_env(gpu: &Gpu, path: &Path) -> anyhow::Result<Env> {
    let img = image::open(path)?.into_rgb32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut px: Vec<[f32; 4]> = img.pixels().map(|p| [p[0], p[1], p[2], 1.0]).collect();
    let lum = |p: &[f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let solid = |y: usize| (2.0 * PI / w as f32) * (PI / h as f32) * (PI * (y as f32 + 0.5) / h as f32).sin();

    let (imax, _) = px.iter().enumerate().fold((0, 0.0f32), |a, (i, p)| if lum(p) > a.1 { (i, lum(p)) } else { a });
    let sun_c = dir_of((imax % w) as f32 / w as f32 + 0.5 / w as f32, (imax / w) as f32 / h as f32 + 0.5 / h as f32);
    let mut ring: Vec<f32> = Vec::new();
    let mut ring_col = [0f32; 3];
    let mut candidates = Vec::new();
    let span = (w as f32 * 6.0 / 360.0) as i64 + 2;
    let (cx, cy) = ((imax % w) as i64, (imax / w) as i64);
    for yy in (cy - span).max(0)..(cy + span).min(h as i64) {
        for xx in cx - span..cx + span {
            let x = xx.rem_euclid(w as i64) as usize;
            let y = yy as usize;
            let d = dir_of((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let ang = d.dot(sun_c).clamp(-1.0, 1.0).acos().to_degrees();
            let i = y * w + x;
            if ang < 3.0 {
                candidates.push(i);
            } else if ang < 4.5 {
                ring.push(lum(&px[i]));
                for k in 0..3 {
                    ring_col[k] += px[i][k];
                }
            }
        }
    }
    ring.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ring_med = ring.get(ring.len() / 2).copied().unwrap_or(1.0);
    let rn = ring.len().max(1) as f32;
    let fill = [ring_col[0] / rn, ring_col[1] / rn, ring_col[2] / rn, 1.0];
    let mut sun_irr = Vec3::ZERO;
    let mut sun_dir = Vec3::ZERO;
    for &i in &candidates {
        let l = lum(&px[i]);
        if l > ring_med * 2.5 {
            let y = i / w;
            let d = dir_of(((i % w) as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let excess = Vec3::new(px[i][0] - fill[0], px[i][1] - fill[1], px[i][2] - fill[2]).max(Vec3::ZERO);
            sun_irr += excess * solid(y);
            sun_dir += d * lum(&[excess.x, excess.y, excess.z, 0.0]) * solid(y);
            px[i] = fill;
        }
    }
    let sun_dir_env = if sun_dir.length() > 0.0 { sun_dir.normalize() } else { Vec3::new(0.3, 0.8, 0.2).normalize() };

    let mut sh = [[0f32; 4]; 9];
    let stride = 4;
    let mut sky_up = 0.0;
    for y in (0..h).step_by(stride) {
        for x in (0..w).step_by(stride) {
            let d = dir_of((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let p = px[y * w + x];
            let dw = solid(y) * (stride * stride) as f32;
            let b = sh_basis(d);
            for k in 0..9 {
                for c in 0..3 {
                    sh[k][c] += p[c] * b[k] * dw;
                }
            }
            if d.y > 0.0 {
                sky_up += lum(&p) * d.y * dw;
            }
        }
    }
    let band = [PI, 2.0 * PI / 3.0, 2.0 * PI / 3.0, 2.0 * PI / 3.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0];
    for k in 0..9 {
        for c in 0..3 {
            sh[k][c] *= band[k];
        }
    }

    let mut levels = vec![(px, w, h)];
    while levels.last().unwrap().1 > 1 || levels.last().unwrap().2 > 1 {
        let (p, lw, lh) = levels.last().unwrap();
        let next = downsample(p, *lw, *lh);
        levels.push(next);
    }
    let tex = texture(
        gpu,
        "env",
        (w as u32, h as u32, 1),
        wgpu::TextureDimension::D2,
        wgpu::TextureFormat::Rgba16Float,
        levels.len() as u32,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    for (mip, (p, lw, lh)) in levels.iter().enumerate() {
        let data: Vec<half::f16> = p.iter().flat_map(|c| c.iter().map(|&v| half::f16::from_f32(v.min(60000.0)))).collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(*lw as u32 * 8), rows_per_image: Some(*lh as u32) },
            wgpu::Extent3d { width: *lw as u32, height: *lh as u32, depth_or_array_layers: 1 },
        );
    }
    Ok(Env {
        view: tex.create_view(&Default::default()),
        sh,
        sun_dir_env,
        sun_irr,
        sky_irr_up: sky_up,
    })
}

fn rgba8_mips(gpu: &Gpu, label: &str, img: image::RgbaImage, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let (w, h) = img.dimensions();
    let mips = 32 - w.max(h).leading_zeros();
    let tex = texture(
        gpu,
        label,
        (w, h, 1),
        wgpu::TextureDimension::D2,
        format,
        mips,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let mut cur = img;
    for mip in 0..mips {
        let (lw, lh) = cur.dimensions();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            cur.as_raw(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lw * 4), rows_per_image: Some(lh) },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
        if mip + 1 < mips {
            cur = image::imageops::resize(&cur, (lw / 2).max(1), (lh / 2).max(1), image::imageops::FilterType::Triangle);
        }
    }
    tex.create_view(&Default::default())
}

pub fn load_ground(gpu: &Gpu, dir: &Path) -> anyhow::Result<Ground> {
    let diff = image::open(dir.join("ground_diff.jpg"))?.into_rgba8();
    let nor = image::open(dir.join("ground_nor_gl.jpg"))?.into_rgb8();
    let rough = image::open(dir.join("ground_rough.jpg"))?.into_luma8();
    let ao = image::open(dir.join("ground_ao.jpg"))?.into_luma8();
    let (w, h) = nor.dimensions();
    let mut packed = image::RgbaImage::new(w, h);
    for (x, y, p) in packed.enumerate_pixels_mut() {
        let n = nor.get_pixel(x, y);
        *p = image::Rgba([n[0], n[1], rough.get_pixel(x, y)[0], ao.get_pixel(x, y)[0]]);
    }
    Ok(Ground {
        albedo: rgba8_mips(gpu, "ground albedo", diff, wgpu::TextureFormat::Rgba8UnormSrgb),
        detail: rgba8_mips(gpu, "ground detail", packed, wgpu::TextureFormat::Rgba8Unorm),
    })
}
