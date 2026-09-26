struct Bullet {
    pos: vec4f,   // tip, w = radius (0 = no bullet)
    dir: vec4f,   // unit axis towards the tip, w = length
    look: vec4f,  // x nose (0 round, 1 spitzer, 2 flat, 3 sphere), y roll angle, z rifling turns per metre, w metal (0 lead, 1 copper, 2 steel)
}

struct Frame {
    cam_pos: vec4f,     // w = tan(fov_y / 2)
    cam_fwd: vec4f,     // w = aspect
    cam_right: vec4f,
    cam_up: vec4f,
    res: vec4f,         // x width, y height, z accumulated frames, w pixel angle
    jitter: vec4f,      // xy subpixel jitter, z seed, w background blur lod
    sun_dir: vec4f,     // xyz towards the sun, w = sun angular radius
    sun_col: vec4f,     // rgb irradiance at normal incidence, w = env rotation (rad)
    sky: vec4f,         // x sky intensity, y exposure, z bloom, w ground tile size (m)
    sh: array<vec4f, 9>,
    knot: vec4f,        // w = knot radius
    rubber: vec4f,      // rgb absorption per mm, w = unstretched thickness (mm)
    rubber2: vec4f,     // rgb diffuse albedo, w = scattering per mm
    water: vec4f,       // x iso, y particle spacing, z kernel radius, w min cell
    wet: vec4f,         // x half size, y resolution, z string top, w string radius
    flags: vec4u,       // x caustics, y max bounces, z mist, w triangle count
    counts: vec4u,      // x particles, y spray capacity, z membrane vertices, w photon grid size
    time: vec4f,        // x sim time, y shutter, z caustic frames, w density normalisation
    lens: vec4f,        // xyz ray direction shift for the current aperture sample
    bullets: array<Bullet, 4>,
}

// Sparse water density over the whole sim domain: cell.w = kernel radius, dims.xyz = bricks
// per axis, dims.w = any water. The coarse smoothing grid follows the water (s_origin.w =
// cell). lo/hi bound the water particles.
struct WGrid { origin: vec4f, cell: vec4f, dims: vec4u, s_origin: vec4f, s_dims: vec4u, lo: vec4f, hi: vec4f }
struct UGrid { origin: vec4f, dims: vec4u }   // uniform cells, origin.w = cell size
struct CMap { center: vec4f, u_axis: vec4f, v_axis: vec4f, ground: vec4f }
// The shown state from the keyframe cache. Particle vel.w flags a live water particle,
// Spray pos.w is the radius (0 = gone) and vel.w the mist weight (1 = spray drop, negative
// while still part of the crown sheet).
struct Particle { pos: vec4f, vel: vec4f }
struct Spray { pos: vec4f, vel: vec4f }
struct Tri { v0: u32, v1: u32, v2: u32, rest_area: f32, e0: u32, e1: u32, e2: u32, pad: u32 }

// Water bricks are 8^3 cells stored as 9^3 texels: the extra texel on the + side of each
// axis repeats the neighbour's first cell so filtering never has to cross into another brick.
const ATLAS_B: u32 = 32u;
const ATLAS: f32 = 288.0;
const BRICK_MARKED: u32 = 0xffffffffu;
const FIXP: f32 = 10000.0;
const RHO_W: f32 = 1000.0;

