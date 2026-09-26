//! Interactive viewer: winit window, egui controls, orbit camera and timeline.

use crate::capture::{assets_dir, offscreen, read_image, save_png, OFFSCREEN_FORMAT};
use crate::env::{load_env, load_ground};
use crate::gpu::Gpu;
use crate::params::{LookParams, SimParams, BULLETS, MAX_SHOTS};
use crate::presets::{self, Preset};
use crate::sim::ShotRun;
use crate::cache::{Cache, T_END};
use crate::render::{Camera, Renderer, MAX_ACCUM};
use egui::{Color32, Key, Sense};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// GPU milliseconds of a 100 Hz frame the app uses. The simulation gets what rendering
/// leaves, so a still, converged view bakes fastest. The rest is headroom for the
/// compositor, so the GPU isn't pinned.
const FRAME_GPU_MS: f64 = 8.5;
/// The simulation's share while the view is busy rendering and playback is well behind it.
const BAKE_MIN_MS: f64 = 3.0;
/// No frame for this long means the compositor isn't showing the window.
const HIDDEN_AFTER: Duration = Duration::from_millis(50);

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App { state: None, error: None };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

struct App {
    state: Option<State>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(el) {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        let resp = s.egui_state.on_window_event(&s.window, &event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => s.resize(size.width, size.height),
            WindowEvent::RedrawRequested => s.frame(),
            _ => {}
        }
        if resp.repaint {
            s.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let Some(s) = self.state.as_mut() else { return };
        if s.busy() {
            s.window.request_redraw();
        }
        // A window the compositor isn't showing gets no redraws, and the bake normally runs
        // once per frame. Until frames come back it runs on its own wakeups.
        let flow = if s.cache.done() {
            ControlFlow::Wait
        } else if s.last_frame.elapsed() > HIDDEN_AFTER {
            s.bake_hidden();
            ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(2))
        } else {
            ControlFlow::WaitUntil(Instant::now() + HIDDEN_AFTER)
        };
        el.set_control_flow(flow);
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_rend: egui_wgpu::Renderer,

    cache: Cache,
    presets: Vec<Preset>,
    sim_edit: SimParams,
    look: LookParams,
    look_used: LookParams,
    cam: Camera,
    playing: bool,
    /// Simulated seconds per real second.
    speed: f64,
    /// Slow the first milliseconds down further, where everything happens.
    ramp: bool,
    t_view: f64,
    /// Simulation steps per frame, sized to fill FRAME_GPU_MS after rendering.
    bake_steps: usize,
    steps_this_frame: usize,
    /// Steps in the frame whose GPU timings are being read back.
    timed_steps: usize,
    render_gpu_ms: f64,
    last_frame: Instant,
    frame_ms: f64,
    message: String,
    screenshot: bool,
}

impl State {
    fn new(el: &ActiveEventLoop) -> anyhow::Result<Self> {
        let attrs = Window::default_attributes().with_title("natray").with_inner_size(winit::dpi::LogicalSize::new(1600.0, 950.0));
        let window = Arc::new(el.create_window(attrs)?);
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(el.owned_display_handle())));
        let surface = instance.create_surface(window.clone())?;
        let gpu = Gpu::new(Some(&surface), &instance)?;
        let caps = surface.get_capabilities(&gpu.adapter);
        // The composite pass writes display-encoded values itself, so no sRGB format.
        let format = [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm]
            .into_iter()
            .find(|f| caps.formats.contains(f))
            .ok_or_else(|| anyhow::anyhow!("no 8-bit linear surface format in {:?}", caps.formats))?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let assets = assets_dir();
        let env = load_env(&gpu, &assets.join("env.hdr"))?;
        let ground = load_ground(&gpu, &assets)?;
        let sim_edit = SimParams::default();
        let cache = Cache::new(&gpu, &sim_edit);
        let mut renderer = Renderer::new(&gpu, env, ground);
        renderer.attach(&gpu, &cache);
        let cam = Camera { target: cache.sim.scene.center, yaw: -35.0, pitch: 8.0, dist: 0.9 };

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_rend = egui_wgpu::Renderer::new(&gpu.device, format, Default::default());
        let look = LookParams::default();

        Ok(Self {
            window,
            surface,
            config,
            gpu,
            renderer,
            egui_ctx,
            egui_state,
            egui_rend,
            t_view: cache.sim.t_start,
            cache,
            presets: presets::all(),
            sim_edit,
            look_used: look.clone(),
            look,
            cam,
            playing: true,
            speed: 1e-3,
            ramp: true,
            bake_steps: 2,
            steps_this_frame: 0,
            timed_steps: 0,
            render_gpu_ms: 0.0,
            last_frame: Instant::now(),
            frame_ms: 16.0,
            message: String::new(),
            screenshot: false,
        })
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.gpu.device, &self.config);
        self.window.request_redraw();
    }

    /// Nothing renders while hidden, so the simulation gets all of FRAME_GPU_MS. Waiting for
    /// it paces the wakeups and times the steps.
    fn bake_hidden(&mut self) {
        // Keeps the timer's marks from piling up while no frame reads them.
        self.gpu.timer.borrow_mut().begin_frame(&self.gpu.device);
        let t0 = Instant::now();
        let n = self.cache.bake(&self.gpu, self.bake_steps);
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let per_step = t0.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64;
        self.bake_steps = ((FRAME_GPU_MS / per_step) as usize).clamp(1, 256);
    }

    fn busy(&self) -> bool {
        self.playing || !self.cache.done() || !self.renderer.converged()
    }

    /// Plays at the chosen speed, times the ramp early on: 1x until 10 ms, then growing
    /// with t up to 25x, so the pop is slow and the fall doesn't drag.
    fn rate(&self) -> f64 {
        let ramp = if self.ramp { (self.t_view / 0.01).clamp(1.0, 25.0) } else { 1.0 };
        self.speed * ramp
    }

    fn seek(&mut self, t: f64) {
        self.t_view = t.clamp(self.cache.sim.t_start, T_END);
    }

    fn waiting_for_sim(&self) -> bool {
        !self.cache.done() && self.t_view >= self.cache.end()
    }

    fn advance(&mut self, wall_dt: f64) {
        if self.gpu.timer.borrow_mut().begin_frame(&self.gpu.device) {
            let span = |l| self.gpu.span_ms(l).unwrap_or(0.0);
            let sim_ms = span("sim") + span("pack");
            self.render_gpu_ms =
                self.gpu.timer.borrow().spans.iter().filter(|(l, _)| !matches!(*l, "sim" | "pack")).map(|s| s.1).sum();
            if self.timed_steps > 0 && sim_ms > 0.0 {
                let per_step = sim_ms / self.timed_steps as f64;
                // About to stall at the end of the simulation: a lower frame rate beats
                // stop-and-go, so the simulation gets the whole budget.
                let catching_up = self.playing && (self.cache.end() - self.t_view) / self.rate() < 1.0;
                let floor = if catching_up { FRAME_GPU_MS } else { BAKE_MIN_MS };
                let budget = (FRAME_GPU_MS - self.render_gpu_ms).max(floor);
                self.bake_steps = ((budget / per_step) as usize).clamp(1, 256);
            }
        }
        self.steps_this_frame = self.cache.bake(&self.gpu, self.bake_steps);
        if self.playing {
            // Playback can't run ahead of the simulation, it waits at the newest keyframe.
            self.t_view = (self.t_view + self.rate() * wall_dt).min(self.cache.end().max(self.t_view));
            if self.t_view >= T_END || (self.cache.done() && self.t_view >= self.cache.end()) {
                self.playing = false;
            }
        }
        if self.cache.show(&self.gpu, self.t_view) {
            self.renderer.reset_accum();
        }
        // At the bake front the next keyframe replaces this one within a few frames, so
        // refining it only takes GPU time from the simulation. One frame looks the same as
        // playback does.
        self.renderer.accum_limit = if self.waiting_for_sim() { 1 } else { MAX_ACCUM };
    }

    fn apply_edits(&mut self, pointer_down: bool) {
        // Rebuilding takes a moment, so wait until the slider is released.
        if self.sim_edit != self.cache.sim.params && !pointer_down {
            self.cache.rebuild(&self.gpu, &self.sim_edit);
            self.renderer.attach(&self.gpu, &self.cache);
            self.cam.target = self.cache.sim.scene.center;
            self.t_view = self.t_view.max(self.cache.sim.t_start);
        }
        if self.look != self.look_used {
            self.renderer.reset_accum();
            self.look_used = self.look.clone();
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let wall_dt = (now - self.last_frame).as_secs_f64().min(0.1);
        self.last_frame = now;
        self.frame_ms += (wall_dt * 1e3 - self.frame_ms) * 0.1;
        self.advance(wall_dt);

        let input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone();
        let mut viewport = egui::Rect::NOTHING;
        let out = ctx.run_ui(input, |ui| viewport = self.ui(ui));
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        self.apply_edits(pointer_down);
        self.egui_state.handle_platform_output(&self.window, out.platform_output);

        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = tex.texture.create_view(&Default::default());
        let ppp = out.pixels_per_point;
        let vp = [
            (viewport.min.x * ppp).round(),
            (viewport.min.y * ppp).round(),
            (viewport.width() * ppp).round().max(8.0),
            (viewport.height() * ppp).round().max(8.0),
        ];
        let scale = self.look.render_scale;
        let rsize = ((vp[2] * scale) as u32, (vp[3] * scale) as u32);
        let bg = wgpu::Color { r: 0.01, g: 0.01, b: 0.012, a: 1.0 };
        self.renderer.render(&self.gpu, &self.cache, &self.cam, &self.look, rsize, &view, self.config.format, vp, bg);
        if self.screenshot {
            self.screenshot = false;
            self.message = match self.save_screenshot(rsize) {
                Ok(p) => format!("saved {p}"),
                Err(e) => format!("screenshot failed: {e}"),
            };
        }

        let prims = ctx.tessellate(out.shapes, ppp);
        for (id, delta) in &out.textures_delta.set {
            self.egui_rend.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: ppp };
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let extra = self.egui_rend.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &prims, &sd);
        {
            let rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.egui_rend.render(&mut rp.forget_lifetime(), &prims, &sd);
        }
        if self.gpu.timer.borrow_mut().resolve(&mut enc) {
            self.timed_steps = self.steps_this_frame;
        }
        self.gpu.queue.submit(extra.into_iter().chain([enc.finish()]));
        self.gpu.timer.borrow_mut().submitted();
        for id in &out.textures_delta.free {
            self.egui_rend.free_texture(id);
        }
        self.window.pre_present_notify();
        tex.present();
    }

    fn save_screenshot(&mut self, size: (u32, u32)) -> anyhow::Result<String> {
        let tex = offscreen(&self.gpu, size);
        let view = tex.create_view(&Default::default());
        let rect = [0.0, 0.0, size.0 as f32, size.1 as f32];
        self.renderer.render(&self.gpu, &self.cache, &self.cam, &self.look, size, &view, OFFSCREEN_FORMAT, rect, wgpu::Color::BLACK);
        std::fs::create_dir_all("captures")?;
        let path = format!("captures/natray_{:.3}ms_{}.png", self.cache.shown * 1e3, chrono_stamp());
        save_png(read_image(&self.gpu, &tex, size), std::path::Path::new(&path))?;
        Ok(path)
    }

    /// Builds the side panel and returns the 3D viewport rect in points.
    fn ui(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        self.shortcuts(ui);
        egui::Panel::right("controls").resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.presets_ui(ui);
                ui.separator();
                self.playback_ui(ui);
                ui.separator();
                self.sim_ui(ui);
                ui.separator();
                self.look_ui(ui);
            });
        });
        egui::CentralPanel::no_frame()
            .show(ui, |ui| {
                let rect = ui.max_rect();
                let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                self.camera_input(ui, &resp);
                self.hud(ui, rect);
                rect
            })
            .inner
    }

    fn shortcuts(&mut self, ui: &egui::Ui) {
        let (space, restart, back, fwd, shot, frame) = ui.input(|i| {
            (
                i.key_pressed(Key::Space),
                i.key_pressed(Key::R),
                i.key_pressed(Key::ArrowLeft),
                i.key_pressed(Key::ArrowRight),
                i.key_pressed(Key::F12) || i.key_pressed(Key::P),
                i.key_pressed(Key::F),
            )
        });
        if space {
            self.playing = !self.playing;
        }
        if restart {
            self.seek(self.cache.sim.t_start);
            self.playing = true;
        }
        let step = self.step_size();
        if back {
            self.playing = false;
            self.seek(self.t_view - step);
        }
        if fwd {
            self.playing = false;
            self.seek(self.t_view + step);
        }
        if shot {
            self.screenshot = true;
        }
        if frame {
            self.cam.target = self.cache.sim.scene.center;
        }
    }

    /// Frame step for the arrow keys: 0.1 ms during the pop, coarser later.
    fn step_size(&self) -> f64 {
        if self.t_view < 0.01 {
            1e-4
        } else if self.t_view < 0.1 {
            1e-3
        } else {
            5e-3
        }
    }

    fn camera_input(&mut self, ui: &egui::Ui, resp: &egui::Response) {
        let before = self.cam;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.cam.yaw -= d.x * 0.25;
            self.cam.pitch = (self.cam.pitch + d.y * 0.25).clamp(-10.0, 89.0);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            let fwd = (self.cam.target - self.cam.position()).normalize();
            let right = fwd.cross(glam::Vec3::Y).normalize();
            let up = right.cross(fwd);
            let k = self.cam.dist * 0.0015;
            self.cam.target += (-right * d.x + up * d.y) * k;
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta().y);
            self.cam.dist = (self.cam.dist * (-scroll * 0.002).exp()).clamp(0.12, 6.0);
        }
        if self.cam != before {
            self.renderer.reset_view_accum();
        }
    }

    fn hud(&self, ui: &egui::Ui, rect: egui::Rect) {
        let state = if self.playing && self.waiting_for_sim() {
            "waiting for the simulation".to_string()
        } else if self.playing {
            format!("{:.0}x slow motion", 1.0 / self.rate())
        } else if self.waiting_for_sim() {
            format!("simulating to {:.2} ms", self.t_view * 1e3)
        } else {
            format!("paused, {} / {} samples", self.renderer.accum_frames.min(MAX_ACCUM), MAX_ACCUM)
        };
        let text = format!(
            "t = {:.3} ms   {}   {:.0} fps   render {:.1} ms GPU\n{}",
            self.cache.shown.max(0.0) * 1e3,
            state,
            1e3 / self.frame_ms.max(1.0),
            self.render_gpu_ms,
            self.message
        );
        ui.painter().text(
            rect.min + egui::vec2(12.0, 10.0),
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::monospace(13.0),
            Color32::from_white_alpha(220),
        );
    }

    fn presets_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Scenes");
        let mut picked = None;
        ui.horizontal_wrapped(|ui| {
            for (i, p) in self.presets.iter().enumerate() {
                if ui.button(p.name).on_hover_text(p.about).clicked() {
                    picked = Some(i);
                }
            }
        });
        if let Some(i) = picked {
            self.load_preset(i);
        }
    }

    fn load_preset(&mut self, i: usize) {
        let p = &self.presets[i];
        self.sim_edit = p.sim.clone();
        self.look = p.look.clone();
        self.message = p.about.to_string();
        let v = p.view;
        self.apply_edits(false);
        self.cam = Camera { target: self.cache.sim.scene.center, yaw: v.yaw, pitch: v.pitch, dist: v.dist };
        self.renderer.reset_view_accum();
        self.seek(self.cache.sim.t_start);
        self.playing = true;
    }

    fn playback_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playback");
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("Restart").clicked() {
                self.seek(self.cache.sim.t_start);
                self.playing = true;
            }
            if ui.button("Screenshot").clicked() {
                self.screenshot = true;
            }
        });
        let mut ms = (self.t_view * 1e3).max(0.0);
        let r = ui.add(
            egui::Slider::new(&mut ms, 0.0..=T_END * 1e3)
                .logarithmic(true)
                .smallest_positive(0.05)
                .text("time")
                .suffix(" ms"),
        );
        if r.changed() {
            self.playing = false;
            self.seek(ms * 1e-3);
        }
        let mb = |b: u64| b as f64 / 1e6;
        let cached = if self.cache.full() {
            format!("cache full at {:.0} ms ({:.0} MB)", self.cache.end() * 1e3, mb(self.cache.pool_bytes()))
        } else if self.cache.done() {
            format!("simulated, {} keyframes, {:.0} of {:.0} MB", self.cache.key_count(), mb(self.cache.used_bytes()), mb(self.cache.pool_bytes()))
        } else {
            format!("simulating: {:.1} of {:.0} ms", self.cache.end() * 1e3, T_END * 1e3)
        };
        ui.add(egui::ProgressBar::new((self.cache.end() / T_END) as f32).text(cached));
        let mut slow = 1.0 / self.speed;
        if ui.add(egui::Slider::new(&mut slow, 10.0..=20000.0).logarithmic(true).text("slow motion").suffix("x")).changed() {
            self.speed = 1.0 / slow;
        }
        ui.checkbox(&mut self.ramp, "Speed up after the pop");
        ui.label("Space play/pause, arrows step, R restart, F refocus, P or F12 screenshot. Drag to orbit, right-drag to pan, scroll to zoom.");
    }

    fn sim_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Simulation");
        ui.label("Applied when you release the slider. The run is re-simulated.");
        // Only while the edits match what is simulated.
        let reports: Vec<String> = if self.sim_edit == self.cache.sim.params { self.cache.sim.shots.iter().map(shot_report).collect() } else { Vec::new() };
        let p = &mut self.sim_edit;
        let n = p.shots.len();
        let mut remove = None;
        for (i, s) in p.shots.iter_mut().enumerate() {
            egui::CollapsingHeader::new(format!("Shot {}", i + 1)).default_open(i == 0).show(ui, |ui| {
                egui::ComboBox::new(("bullet", i), "bullet").selected_text(s.bullet.name).show_ui(ui, |ui| {
                    for b in BULLETS {
                        if ui.selectable_label(s.bullet == b, b.name).clicked() {
                            s.set_bullet(b);
                        }
                    }
                });
                ui.add(egui::Slider::new(&mut s.speed, 50.0..=1200.0).text("speed at impact").suffix(" m/s"));
                ui.add(egui::Slider::new(&mut s.caliber_mm, 4.0..=12.0).text("caliber").suffix(" mm"));
                ui.add(egui::Slider::new(&mut s.mass_g, 0.2..=20.0).logarithmic(true).text("bullet mass").suffix(" g"));
                ui.add(egui::Slider::new(&mut s.spin_rpm, 0.0..=400_000.0).text("spin").suffix(" rpm"));
                ui.add(egui::Slider::new(&mut s.range_m, 1.0..=100.0).logarithmic(true).text("range").suffix(" m"));
                ui.add(egui::Slider::new(&mut s.muzzle_yaw_deg, 0.0..=10.0).text("yaw at the muzzle").suffix("°"));
                ui.small("Spin keeps a bullet pointed in air. In water no spin is enough: it yaws, a long one tumbles, and that widens the cavity and the exit hole.");
                ui.add(egui::Slider::new(&mut s.azimuth_deg, -180.0..=180.0).text("shot from, around").suffix("°"));
                ui.add(egui::Slider::new(&mut s.elevation_deg, -60.0..=60.0).text("shot from, above").suffix("°"));
                ui.add(egui::Slider::new(&mut s.impact_offset_cm, -5.0..=5.0).text("impact height offset").suffix(" cm"));
                ui.add(egui::Slider::new(&mut s.impact_side_cm, -5.0..=5.0).text("impact side offset").suffix(" cm"));
                ui.add(egui::Slider::new(&mut s.delay_ms, 0.0..=200.0).logarithmic(true).smallest_positive(0.01).text("arrives after").suffix(" ms"));
                if let Some(r) = reports.get(i) {
                    ui.small(r);
                }
                if n > 1 && ui.button("Remove shot").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            p.shots.remove(i);
        }
        if n < MAX_SHOTS && ui.button("Add shot").clicked() {
            p.add_shot();
        }
        ui.separator();
        ui.add(egui::Slider::new(&mut p.balloon_diameter_cm, 8.0..=24.0).text("balloon diameter").suffix(" cm"));
        ui.add(egui::Slider::new(&mut p.balloon_height, 0.25..=1.2).text("hanging height").suffix(" m"));
        ui.add(egui::Slider::new(&mut p.rubber_tension, 60.0..=300.0).text("rubber tension").suffix(" N/m"));
        ui.add(egui::Slider::new(&mut p.rubber_strength, 1.05..=2.5).text("rubber strength"));
        ui.add(egui::Slider::new(&mut p.rubber_thickness_mm, 0.1..=0.6).text("rubber thickness").suffix(" mm"));
        ui.add(egui::Slider::new(&mut p.gravity, 0.0..=25.0).text("gravity").suffix(" m/s²"));
        ui.add(egui::Slider::new(&mut p.spray_amount, 0.0..=3.0).text("spray amount"));
        ui.add(egui::Slider::new(&mut p.mist_amount, 0.0..=3.0).text("mist amount"));
        ui.add(egui::Slider::new(&mut p.resolution, 32..=96).text("water resolution"));
        if ui.button("Defaults").clicked() {
            *p = SimParams::default();
        }
    }

    fn look_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Look");
        let l = &mut self.look;
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut l.balloon_color);
            ui.label("balloon colour");
        });
        ui.add(egui::Slider::new(&mut l.sun_intensity, 0.0..=3.0).text("sun"));
        ui.add(egui::Slider::new(&mut l.sky_intensity, 0.0..=3.0).text("sky"));
        ui.add(egui::Slider::new(&mut l.env_rotation_deg, 0.0..=360.0).text("sun direction").suffix("°"));
        ui.add(egui::Slider::new(&mut l.exposure_ev, -4.0..=4.0).text("exposure").suffix(" EV"));
        ui.checkbox(&mut l.dof, "depth of field");
        ui.add_enabled(l.dof, egui::Slider::new(&mut l.f_stop, 1.4..=22.0).logarithmic(true).text("aperture f/"));
        ui.add(egui::Slider::new(&mut l.focal_mm, 18.0..=200.0).logarithmic(true).text("focal length").suffix(" mm"));
        ui.add(egui::Slider::new(&mut l.shutter_us, 1.0..=2000.0).logarithmic(true).text("shutter").suffix(" µs"));
        ui.add(egui::Slider::new(&mut l.bloom, 0.0..=0.2).text("bloom"));
        ui.add(egui::Slider::new(&mut l.render_scale, 0.25..=1.0).text("render scale"));
        ui.add(egui::Slider::new(&mut l.max_bounces, 1..=16).text("water bounces"));
        ui.checkbox(&mut l.caustics, "caustics");
        ui.checkbox(&mut l.mist, "mist");
        if ui.button("Defaults").clicked() {
            *l = LookParams::default();
        }
    }
}

fn shot_report(s: &ShotRun) -> String {
    let r = &s.flight.report;
    let air = if r.sg_air.is_infinite() {
        "A ball has no axis to tip over".to_string()
    } else if r.sg_air < 1.0 {
        format!("Unstable in air (Sg {:.2}), {:.0}° of yaw on arrival", r.sg_air, r.yaw_impact_deg)
    } else {
        format!("Stable in air (Sg {:.1}), {:.1}° of yaw on arrival", r.sg_air, r.yaw_impact_deg)
    };
    let water = if !r.done {
        "gets to the water later in the run".to_string()
    } else if s.flight.path.is_none() {
        "misses the water".to_string()
    } else if r.stopped {
        format!("yaws up to {:.0}° and stops inside", r.max_yaw_water_deg)
    } else {
        let tumbles = if r.max_yaw_water_deg > 90.0 { ", tumbling" } else { "" };
        format!("yaws up to {:.0}° in the water{tumbles}, comes out at {:.0} m/s", r.max_yaw_water_deg, r.v_exit)
    };
    format!("{air}; {water}.")
}

fn chrono_stamp() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
