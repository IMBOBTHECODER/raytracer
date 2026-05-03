import os
import pyassimp
from config import Config
import numpy as np
import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

tri_v0           = None
tri_v1           = None
tri_v2           = None
tri_colour       = None
tri_roughness    = None
tri_metalness    = None
tri_transmission = None
tri_emission     = None
tri_normal       = None
tri_n0           = None
tri_n1           = None
tri_n2           = None
tri_has_smooth   = None
tri_uv0          = None
tri_uv1          = None
tri_uv2          = None
tri_tex_id       = None

tex_atlas  = None
n_textures = 0
TEX_SIZE   = 4096

# BVH node fields (populated by build())
bvh_bbox_min  = None  # vec3 per node — AABB min corner
bvh_bbox_max  = None  # vec3 per node — AABB max corner
bvh_left      = None  # int per node  — left child index (-1 if leaf)
bvh_right     = None  # int per node  — right child index (-1 if leaf)
bvh_tri_start = None  # int per node  — first index into bvh_tri_indices (leaf only)
bvh_tri_end   = None  # int per node  — one-past-last index into bvh_tri_indices (leaf only)
bvh_tri_indices = None  # flat list: maps node's [tri_start..tri_end) → triangle index

BVH_STACK_SIZE = 64   # max tree depth * 2; increase if your BVH is very deep

# Plane fields (populated by build())
plane_point    = None
plane_normal   = None
plane_colour   = None
plane_material = None
plane_metal_fuzz       = None
plane_refraction_index = None
plane_emission         = None
n_planes = 0

# Sphere fields (populated by build())
sphere_center  = None
sphere_radius  = None
sphere_colour  = None
sphere_material = None
sphere_metal_fuzz       = None
sphere_refraction_index = None
sphere_emission         = None
n_spheres = 0

framebuffer = ti.Vector.field(3, dtype=ti.f32, shape=(Config.img_height, Config.img_width))

# ── Owen-Scrambled Sobol ──────────────────────────────────────────────────────
# Replaces ti.random() calls throughout. Dim layout per bounce (DIMS_PER_BOUNCE=5):
#   base+0: material branch   (was `r` in scatter)
#   base+1: scatter u
#   base+2: scatter v
#   base+3: NEE light sample u
#   base+4: NEE light sample v
# Plus 2 leading dims for depth-of-field (dims 0 and 1).
# AA jitter reuses bounce-0 scatter dims — no conflict since those are consumed
# before any surface is hit.
DIMS_PER_BOUNCE = 5
SOBOL_DIMS      = 2 + DIMS_PER_BOUNCE * Config.depth_limit
SOBOL_BITS      = 32
MAX_SOBOL_DIMS  = 64

DIM_DOF_U       = 0
DIM_DOF_V       = 1
DIM_BOUNCE_BASE = 2   # bounce k uses dims DIM_BOUNCE_BASE + k*DIMS_PER_BOUNCE ..+4
DIM_MAT         = 0   # offset within a bounce block
DIM_SCATTER_U   = 1
DIM_SCATTER_V   = 2
DIM_LIGHT_U     = 3
DIM_LIGHT_V     = 4

# Joe & Kuo direction-number initialisers (first 10 non-trivial dims).
# Full table: https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
_JK_INIT = [
    (1, 0,  [1]),
    (2, 1,  [1, 1]),
    (3, 1,  [1, 1, 1]),
    (3, 2,  [1, 1, 3]),
    (4, 1,  [1, 1, 1, 3]),
    (4, 4,  [1, 1, 3, 1]),
    (5, 2,  [1, 1, 1, 1, 1]),
    (5, 4,  [1, 1, 3, 5, 3]),
    (5, 7,  [1, 1, 1, 5, 5]),
    (5, 11, [1, 1, 3, 3, 1]),
]

def _build_sobol_dirs(max_dims: int, n_bits: int) -> np.ndarray:
    dirs = np.zeros((max_dims, n_bits), dtype=np.uint32)
    # Dim 0 — Van der Corput
    for b in range(n_bits):
        dirs[0, b] = np.uint32(1) << np.uint32(n_bits - 1 - b)
    # Dims 1..len(_JK_INIT) — Joe & Kuo recurrence
    for dim_idx, (s, a, m_init) in enumerate(_JK_INIT):
        d = dim_idx + 1
        if d >= max_dims:
            break
        v = np.zeros(n_bits + 1, dtype=np.uint64)
        for i, mi in enumerate(m_init, start=1):
            v[i] = np.uint64(mi) << np.uint64(n_bits - i)
        for i in range(s + 1, n_bits + 1):
            v[i] = v[i - s] ^ (v[i - s] >> np.uint64(s))
            for k in range(1, s):
                bit_k = (np.uint64(a) >> np.uint64(s - 1 - k)) & np.uint64(1)
                v[i] ^= bit_k * v[i - k]
        for b in range(n_bits):
            dirs[d, b] = np.uint32(v[b + 1] & np.uint64(0xFFFFFFFF))
    # Remaining dims — cycle through known ones
    for d in range(len(_JK_INIT) + 1, max_dims):
        for b in range(n_bits):
            dirs[d, b] = dirs[(d - 1) % (len(_JK_INIT) + 1), b]
    return dirs

def _owen_hash(x: np.uint32, seed: np.uint32) -> np.uint32:
    x    = np.uint32(x)
    seed = np.uint32(seed)
    x ^= x    * np.uint32(0x3d20adea)
    x += seed
    x *= (seed >> np.uint32(2)) | np.uint32(1)
    x ^= x    * np.uint32(0x05526c56)
    x ^= x    * np.uint32(0x53a22864)
    return x

def build_sobol_buffer(spp: int, n_dims: int, frame_seed: int = 0) -> np.ndarray:
    """Build (spp, n_dims) Owen-scrambled Sobol buffer on CPU."""
    dirs    = _build_sobol_dirs(MAX_SOBOL_DIMS, SOBOL_BITS)
    indices = np.arange(spp, dtype=np.uint32)
    buf     = np.empty((spp, n_dims), dtype=np.float32)
    for dim in range(n_dims):
        dim_seed = np.uint32(_owen_hash(np.uint32(dim), np.uint32(frame_seed)))
        raw = np.zeros(spp, dtype=np.uint32)
        for b in range(SOBOL_BITS):
            mask = ((indices >> np.uint32(b)) & np.uint32(1)).astype(bool)
            raw[mask] ^= dirs[dim % MAX_SOBOL_DIMS, b]
        # Vectorised Owen hash (same ops as _owen_hash, applied elementwise)
        x  = raw ^ (raw * np.uint32(0x3d20adea))
        x  = x + dim_seed
        x  = x * ((dim_seed >> np.uint32(2)) | np.uint32(1))
        x  = x ^ (x  * np.uint32(0x05526c56))
        x  = x ^ (x  * np.uint32(0x53a22864))
        buf[:, dim] = x.astype(np.float64) / 4294967296.0
    return buf

# ── Blue Noise Pixel Seeds (void-and-cluster, Ulichney 1993) ─────────────────
BN_TILE  = 64           # tile side length; 64×64 = 4096 unique rank values
BN_CACHE = "bn_tile.npy"  # persisted across runs — generated once, reused forever


def _bn_convolve(mask: np.ndarray, kernel_f: np.ndarray) -> np.ndarray:
    """Toroidal Gaussian convolution using a pre-computed rfft2 kernel."""
    return np.fft.irfft2(
        np.fft.rfft2(mask.astype(np.float32)) * kernel_f,
        s=mask.shape,
    ).astype(np.float32)


def generate_blue_noise_tile(size: int = BN_TILE, sigma: float = 1.5, seed: int = 0) -> np.ndarray:
    """
    Void-and-cluster blue noise.
    Returns a (size, size) uint32 array of ranks in [0, size²).
    Phases 2 and 3 each do size²//2 FFT convolutions on a size×size tile,
    so size=64 finishes in well under a second.
    """
    rng = np.random.default_rng(seed)
    N   = size * size

    # Toroidal Gaussian kernel: wrap distances so the tile tiles seamlessly
    r    = np.minimum(np.arange(size), size - np.arange(size)).astype(np.float32)
    Y, X = np.meshgrid(r, r, indexing="ij")
    kernel_f = np.fft.rfft2(np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2)))

    # Initial binary pattern — exactly N//2 ones placed at random
    flat = np.zeros(N, dtype=np.float32)
    flat[: N // 2] = 1.0
    rng.shuffle(flat)
    mask = flat.reshape(size, size)

    # Phase 1: swap highest-cluster 1 → lowest-void 0, repeat until stable.
    # N//2 iterations is more than enough for a 64×64 tile.
    for _ in range(N // 2):
        e = _bn_convolve(mask, kernel_f).ravel()
        m = mask.ravel()                          # view — writes propagate to mask
        ones  = np.where(m > 0.5)[0]
        voids = np.where(m < 0.5)[0]
        hc = ones [np.argmax(e[ones ])]
        lv = voids[np.argmin(e[voids])]
        m[hc] = 0.0
        m[lv] = 1.0

    ranks = np.empty(N, dtype=np.uint32)
    init  = mask.ravel().copy()                   # converged pattern, flat
    n1    = int(init.sum())                        # number of ones after Phase 1

    # Phase 2: assign ranks [n1-1 .. 0] by peeling off the highest-cluster 1
    w = init.copy()
    for rank in range(n1 - 1, -1, -1):
        e   = _bn_convolve(w.reshape(size, size), kernel_f).ravel()
        idx = np.where(w > 0.5)[0]
        hc  = idx[np.argmax(e[idx])]
        ranks[hc] = rank
        w[hc] = 0.0

    # Phase 3: assign ranks [n1 .. N-1] by filling the lowest-void 0
    w = init.copy()
    for rank in range(n1, N):
        e   = _bn_convolve(w.reshape(size, size), kernel_f).ravel()
        idx = np.where(w < 0.5)[0]
        lv  = idx[np.argmin(e[idx])]
        ranks[lv] = rank
        w[lv] = 1.0

    return ranks.reshape(size, size)


def _load_or_gen_blue_noise() -> np.ndarray:
    """Return a cached blue noise tile, or generate and cache it."""
    if os.path.exists(BN_CACHE):
        tile = np.load(BN_CACHE)
        if tile.shape == (BN_TILE, BN_TILE):
            return tile
    print(f"[Sampler] Generating {BN_TILE}×{BN_TILE} blue noise tile (saved to {BN_CACHE})…")
    tile = generate_blue_noise_tile(BN_TILE)
    np.save(BN_CACHE, tile)
    return tile


# Populated by init_sobol() — call before render().
sobol_buf   = None  # ti.field(f32, shape=(spp, SOBOL_DIMS))
pixel_seeds = None  # ti.field(u32, shape=(H, W))

def init_sobol(spp: int):
    """Call once after scene setup, before the first render() call."""
    global sobol_buf, pixel_seeds

    buf_np = build_sobol_buffer(spp, SOBOL_DIMS)
    sobol_buf = ti.field(dtype=ti.f32, shape=(spp, SOBOL_DIMS))
    sobol_buf.from_numpy(buf_np)

    # Blue noise pixel seeds: tile the rank map and map to full uint32 range.
    # Neighboring pixels get well-separated scrambles → screen-space error is
    # blue-noise distributed (perceptually nicer at low spp).
    bn_tile  = _load_or_gen_blue_noise()
    bn_u32   = (bn_tile.astype(np.float64)
                * (np.float64(0xFFFFFFFF) / float(BN_TILE * BN_TILE - 1))
               ).astype(np.uint32)
    reps_h   = (Config.img_height + BN_TILE - 1) // BN_TILE
    reps_w   = (Config.img_width  + BN_TILE - 1) // BN_TILE
    seeds_np = np.tile(bn_u32, (reps_h, reps_w))[:Config.img_height, :Config.img_width].copy()
    pixel_seeds = ti.field(dtype=ti.u32, shape=(Config.img_height, Config.img_width))
    pixel_seeds.from_numpy(seeds_np)

@ti.func
def sobol(sample_idx: ti.i32, dim: ti.i32, pixel_seed: ti.u32) -> ti.f32:
    """Fetch pre-built Sobol sample and apply a fast per-pixel Owen scramble."""
    raw_u = ti.cast(sobol_buf[sample_idx, dim] * 4294967296.0, ti.u32)
    h  = raw_u ^ pixel_seed
    h ^= h >> 16
    h *= ti.u32(0x85ebca6b)
    h ^= h >> 13
    h *= ti.u32(0xc2b2ae35)
    h ^= h >> 16
    return ti.cast(h, ti.f32) * ti.f32(2.3283064365386963e-10)  # / 2^32

# ─────────────────────────────────────────────────────────────────────────────
# Everything below is identical to the original except:
#   1. sphere_light_sample: two ti.random() → caller-supplied r1, r2 args
#   2. scatter: three ti.random() → caller-supplied r, su1, su2 args
#   3. ray_colour: new sample_idx + pixel_seed params; fetches sobol dims
#   4. render: passes pseed + sample index; stratified jitter uses sobol
# ─────────────────────────────────────────────────────────────────────────────

@ti.func
def lerp(a, b, t):
    # Blend between a and b
    return a + t * (b - a)

@ti.func
def normalize(v):
    n = v.norm()
    return v / (n if n > 1e-8 else 1.0)

@ti.func
def background(direction):
    unit_direction = direction.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    white = tm.vec3(1.0, 1.0, 1.0)
    blue  = tm.vec3(0.5, 0.7, 1.0)
    return (1.0 - t) * white + t * blue

@ti.func
def power_heuristic(pdf_a, pdf_b):
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / (a2 + b2)   # weight for sampler A

def apply_effect(linear: np.ndarray):
    """Apply bloom post-processing."""
    linear = linear.astype(np.float32)

    # Bloom — cinematic glow on bright highlights
    bright_hdr  = np.clip(linear - 0.6, 0, None)          # lower threshold = more surfaces glow
    blur_tight  = cv2.GaussianBlur(bright_hdr, (51,  51),  0)
    blur_mid    = cv2.GaussianBlur(bright_hdr, (101, 101), 0)
    blur_wide   = cv2.GaussianBlur(bright_hdr, (251, 251), 0)
    blur_hdr    = blur_tight * 0.3 + blur_mid * 0.4 + blur_wide * 0.6
    blur_hdr   *= 0.7
    # Cool/blue tint on bloom
    blur_hdr   *= np.array([0.8, 0.9, 1.2], dtype=np.float32)
    bloomed = linear + blur_hdr

    out = (np.clip(bloomed, 0, 1) * 255).astype(np.uint8)
    return out

@ti.func
def aabb_hit(node_idx, ray_o, ray_d, t_min, t_max):
    """Slab test — returns True if ray hits the AABB of this node."""
    for axis in ti.static(range(3)):
        inv_d = 1.0 / ray_d[axis] if ti.abs(ray_d[axis]) > 1e-10 else 1e30
        t0 = (bvh_bbox_min[node_idx][axis] - ray_o[axis]) * inv_d
        t1 = (bvh_bbox_max[node_idx][axis] - ray_o[axis]) * inv_d
        if inv_d < 0:
            t0, t1 = t1, t0
        t_min = tm.max(t_min, t0)
        t_max = tm.min(t_max, t1)
    return t_max > t_min  # check once after all 3 axes

@ti.func
def tri_intersect(tri_idx, ray_o, ray_d, t_min, t_max):
    """Möller-Trumbore intersection. Returns (t, u, v) — t < 0 means no hit."""
    v0 = tri_v0[tri_idx]
    edge1 = tri_v1[tri_idx] - v0
    edge2 = tri_v2[tri_idx] - v0

    h = tm.cross(ray_d, edge2)
    a = tm.dot(edge1, h)

    t, u, v = -1.0, 0.0, 0.0
    if ti.abs(a) > 1e-8:          # not parallel
        f = 1.0 / a
        s = ray_o - v0
        u = f * tm.dot(s, h)
        if u >= 0.0 and u <= 1.0:
            q = tm.cross(s, edge1)
            v = f * tm.dot(ray_d, q)
            if v >= 0.0 and u + v <= 1.0:
                t_hit = f * tm.dot(edge2, q)
                if t_min <= t_hit <= t_max:
                    t = t_hit
    return t, u, v

@ti.func
def bvh_hit(ray_o, ray_d, t_min, t_max):
    """Iterative stack-based BVH traversal. Returns (t, tri_idx, u, v)."""
    best_t   = -1.0
    best_idx = -1
    best_u   = 0.0
    best_v   = 0.0

    # Fixed-size stack — stores node indices to visit
    stack     = ti.Vector([0] * BVH_STACK_SIZE, dt=ti.i32)
    stack_ptr = 0
    stack[stack_ptr] = 0  # start at root (node 0)
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        far = best_t if best_t > 0 else t_max
        if not aabb_hit(node_idx, ray_o, ray_d, t_min, far):
            continue

        if bvh_left[node_idx] == -1:  # leaf node
            for k in range(bvh_tri_start[node_idx], bvh_tri_end[node_idx]):
                tri_idx = bvh_tri_indices[k]
                t, u, v = tri_intersect(tri_idx, ray_o, ray_d, t_min, far)
                if t > 0 and (best_t < 0 or t < best_t):
                    best_t   = t
                    best_idx = tri_idx
                    best_u   = u
                    best_v   = v
                    far = best_t  # tighten bound
        else:
            # push both children — right first so left is processed first
            stack[stack_ptr] = bvh_right[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = bvh_left[node_idx]
            stack_ptr += 1

    return best_t, best_idx, best_u, best_v

@ti.func
def get_normal(tri_idx, ray_d, u, v):
    n = tm.vec3(0.0)
    if tri_has_smooth[tri_idx]:
        n = (1.0 - u - v) * tri_n0[tri_idx] + u * tri_n1[tri_idx] + v * tri_n2[tri_idx]
        n = normalize(n)
    else:
        n = tri_normal[tri_idx]
    # flip if ray hits the back face
    if tm.dot(ray_d, n) > 0:
        n = -n
    return n

@ti.func
def sample_texture(tex_id, uv):
    u  = uv[0] % 1.0
    v  = 1.0 - (uv[1] % 1.0)  # flip V: UV origin is bottom-left, image origin is top-left
    x  = u * (TEX_SIZE - 1)
    y  = v * (TEX_SIZE - 1)
    x0 = int(x);  x1 = min(x0 + 1, TEX_SIZE - 1)
    y0 = int(y);  y1 = min(y0 + 1, TEX_SIZE - 1)
    fx = x - float(x0)
    fy = y - float(y0)
    c00 = tex_atlas[tex_id, y0, x0]
    c10 = tex_atlas[tex_id, y0, x1]
    c01 = tex_atlas[tex_id, y1, x0]
    c11 = tex_atlas[tex_id, y1, x1]
    return (1.0-fx)*(1.0-fy)*c00 + fx*(1.0-fy)*c10 + (1.0-fx)*fy*c01 + fx*fy*c11

@ti.func
def sphere_light_sample(sphere_idx, hit_point, r1: ti.f32, r2: ti.f32):
    """Sample a direction toward sphere light. Returns (dir, pdf); pdf=0 on degenerate.
    r1, r2 replace the two ti.random() calls that were inside this function."""
    center = sphere_center[sphere_idx]
    radius = sphere_radius[sphere_idx]
    to_center = center - hit_point
    dist2 = to_center.norm_sqr()

    light_dir = tm.vec3(0.0)
    pdf = 0.0

    if dist2 > radius * radius:
        sin2_max = radius * radius / dist2
        cos_max  = tm.sqrt(tm.max(0.0, 1.0 - sin2_max))
        solid    = 2.0 * tm.pi * (1.0 - cos_max)
        pdf      = 1.0 / solid

        w = normalize(to_center)
        up = tm.vec3(0.0, 1.0, 0.0)
        if ti.abs(w[1]) > 0.9:
            up = tm.vec3(1.0, 0.0, 0.0)
        uu = normalize(tm.cross(up, w))
        vv = tm.cross(w, uu)

        cos_t = 1.0 - r1 * (1.0 - cos_max)
        sin_t = tm.sqrt(tm.max(0.0, 1.0 - cos_t * cos_t))
        phi   = 2.0 * tm.pi * r2
        light_dir = normalize(sin_t * tm.cos(phi) * uu + sin_t * tm.sin(phi) * vv + cos_t * w)

    return light_dir, pdf


@ti.func
def sphere_light_pdf(sphere_idx, hit_point, direction):
    """PDF of sampling `direction` toward sphere light (0 if direction misses the cone)."""
    center = sphere_center[sphere_idx]
    radius = sphere_radius[sphere_idx]
    to_center = center - hit_point
    dist2 = to_center.norm_sqr()
    pdf = 0.0
    if dist2 > radius * radius:
        sin2_max = radius * radius / dist2
        cos_max  = tm.sqrt(tm.max(0.0, 1.0 - sin2_max))
        solid    = 2.0 * tm.pi * (1.0 - cos_max)
        cos_dir  = tm.dot(direction, normalize(to_center))
        if cos_dir > cos_max:
            pdf = 1.0 / solid
    return pdf


@ti.func
def scatter(colour, roughness, metalness, transmission, ray_d, normal,
            r: ti.f32, su1: ti.f32, su2: ti.f32):
    """Material BRDF — identical to original except the three ti.random() calls
    are replaced by caller-supplied samples r, su1, su2."""
    scatter_dir = tm.vec3(0.0)
    attenuation = colour
    did_scatter = False

    pdf = 0.0

    if r < transmission:  # Glass
        ior = 1.5
        d = ray_d
        front_face = tm.dot(d, normal) < 0
        n = normal if front_face else -normal
        eta = 1.0 / ior if front_face else ior
        cos_theta = tm.min(tm.dot(-d, n), 1.0)
        sin_theta = tm.sqrt(tm.max(0.0, 1.0 - cos_theta ** 2))
        r0 = ((1 - eta) / (1 + eta)) ** 2
        reflectance = r0 + (1 - r0) * (1 - cos_theta) ** 5
        if eta * sin_theta > 1.0 or su1 < reflectance:  # su1 replaces the second ti.random()
            scatter_dir = d - 2 * tm.dot(d, n) * n
        else:
            d_perp     = eta * (d + cos_theta * n)
            d_parallel = -tm.sqrt(ti.abs(1.0 - d_perp.norm_sqr())) * n
            scatter_dir = d_perp + d_parallel
        attenuation = colour
        pdf = 1.0
        did_scatter = True

    elif r < transmission + metalness:  # Metal — uniform sphere fuzz via su1/su2
        d = normalize(ray_d)
        reflected = d - 2 * tm.dot(d, normal) * normal
        fuzz_phi   = 2.0 * tm.pi * su1
        fuzz_cos_t = 1.0 - 2.0 * su2
        fuzz_sin_t = tm.sqrt(tm.max(0.0, 1.0 - fuzz_cos_t * fuzz_cos_t))
        reflected += roughness * tm.vec3(fuzz_sin_t * tm.cos(fuzz_phi),
                                         fuzz_cos_t,
                                         fuzz_sin_t * tm.sin(fuzz_phi))
        did_scatter = tm.dot(reflected, normal) > 0
        pdf = 1.0
        scatter_dir = reflected

    else:  # Diffuse — cosine-weighted hemisphere via su1/su2 (Malley's method)
        diff_phi   = 2.0 * tm.pi * su1
        diff_sin_t = tm.sqrt(su2)
        diff_cos_t = tm.sqrt(1.0 - su2)
        up = tm.vec3(0.0, 1.0, 0.0)
        if ti.abs(normal[1]) > 0.9:
            up = tm.vec3(1.0, 0.0, 0.0)
        tangent   = normalize(tm.cross(up, normal))
        bitangent = tm.cross(normal, tangent)
        scatter_dir = (diff_sin_t * tm.cos(diff_phi) * tangent
                       + diff_sin_t * tm.sin(diff_phi) * bitangent
                       + diff_cos_t * normal)
        pdf = diff_cos_t / tm.pi
        did_scatter = True

    return scatter_dir, attenuation, pdf, did_scatter


@ti.func
def plane_hit(plane_idx, ray_o, ray_d, t_min, t_max):
    """Ray-plane intersection. Returns t, or -1 on miss."""
    n = plane_normal[plane_idx]
    denom = tm.dot(ray_d, n)
    t = -1.0
    if ti.abs(denom) > 1e-6:
        t_hit = tm.dot(plane_point[plane_idx] - ray_o, n) / denom
        if t_min <= t_hit and t_hit <= t_max:
            t = t_hit
    return t


@ti.func
def sphere_hit(sphere_idx, ray_o, ray_d, t_min, t_max):
    """Ray-sphere intersection. Returns t, or -1 on miss."""
    oc = ray_o - sphere_center[sphere_idx]
    a  = tm.dot(ray_d, ray_d)
    b  = 2.0 * tm.dot(oc, ray_d)
    c  = tm.dot(oc, oc) - sphere_radius[sphere_idx] ** 2
    disc = b * b - 4 * a * c
    t = -1.0
    if disc >= 0:
        sqrt_d = tm.sqrt(disc)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)
        if t_min <= t1 and t1 <= t_max:
            t = t1
        elif t_min <= t2 and t2 <= t_max:
            t = t2
    return t


@ti.func
def scene_hit(ray_o, ray_d, t_min, t_max):
    """Check all objects. Returns (t, hit_type, obj_idx, u, v).
    hit_type: -1=miss, 0=triangle, 1=plane, 2=sphere."""
    best_t    = -1.0
    best_type = -1
    best_idx  = -1
    best_u    = 0.0
    best_v    = 0.0

    # Triangles via BVH
    t, tri_idx, u, v = bvh_hit(ray_o, ray_d, t_min, t_max)
    if t > 0:
        best_t = t;  best_type = 0;  best_idx = tri_idx
        best_u = u;  best_v = v

    # Planes
    for i in range(n_planes):
        far = best_t if best_t > 0 else t_max
        t = plane_hit(i, ray_o, ray_d, t_min, far)
        if t > 0 and (best_t < 0 or t < best_t):
            best_t = t;  best_type = 1;  best_idx = i

    # Spheres
    for i in range(n_spheres):
        far = best_t if best_t > 0 else t_max
        t = sphere_hit(i, ray_o, ray_d, t_min, far)
        if t > 0 and (best_t < 0 or t < best_t):
            best_t = t;  best_type = 2;  best_idx = i

    return best_t, best_type, best_idx, best_u, best_v


@ti.func
def ray_colour(ray_o, ray_d, sample_idx: ti.i32, pixel_seed: ti.u32):
    colour     = tm.vec3(0.0)
    throughput = tm.vec3(1.0)
    prev_hit_point = ray_o
    prev_brdf_pdf  = 1.0
    prev_specular  = True  # camera ray has no BRDF; treat like delta — always include emission

    for bounce in range(Config.depth_limit):
        t, hit_type, obj_idx, u, v = scene_hit(ray_o, ray_d, 0.001, 1e30)

        if hit_type == -1:
            colour += throughput * background(ray_d)
            break

        hit_point = ray_o + t * ray_d

        # ── Normal ───────────────────────────────────────────────────────────
        normal = tm.vec3(0.0)
        if hit_type == 0:  # triangle
            normal = get_normal(obj_idx, ray_d, u, v)
        elif hit_type == 1:  # plane
            normal = plane_normal[obj_idx]
            if tm.dot(ray_d, normal) > 0:
                normal = -normal
        else:  # sphere
            normal = normalize(hit_point - sphere_center[obj_idx])
            if tm.dot(ray_d, normal) > 0:
                normal = -normal

        # ── Material lookup ──────────────────────────────────────────────────
        mat_colour    = tm.vec3(0.0)
        mat_roughness = 1.0
        mat_metalness = 0.0
        mat_trans     = 0.0
        mat_emission  = tm.vec3(0.0)

        if hit_type == 0:  # triangle
            tid = tri_tex_id[obj_idx]
            if tid >= 0:
                uv = tri_uv0[obj_idx] * (1.0 - u - v) + tri_uv1[obj_idx] * u + tri_uv2[obj_idx] * v
                mat_colour = sample_texture(tid, uv)
            else:
                mat_colour = tri_colour[obj_idx]
            mat_roughness = tri_roughness[obj_idx]
            mat_metalness = tri_metalness[obj_idx]
            mat_trans     = tri_transmission[obj_idx]
            mat_emission  = tri_emission[obj_idx]
        elif hit_type == 1:  # plane — checkerboard diffuse
            cx = int(hit_point[0]) if hit_point[0] >= 0 else int(hit_point[0]) - 1
            cz = int(hit_point[2]) if hit_point[2] >= 0 else int(hit_point[2]) - 1
            if (cx + cz) % 2 == 0:
                mat_colour = tm.vec3(0.8, 0.8, 0.8)
            else:
                mat_colour = tm.vec3(0.1, 0.1, 0.1)
            smat = plane_material[obj_idx]
            mat_roughness = plane_metal_fuzz[obj_idx] if smat == 1 else 1.0
            mat_metalness = 1.0 if smat == 1 else 0.0
            mat_trans     = 1.0 if smat == 2 else 0.0
            mat_emission  = mat_colour * plane_emission[obj_idx] if smat == 3 else tm.vec3(0.0)
        else:  # sphere
            mat_colour    = sphere_colour[obj_idx]
            smat          = sphere_material[obj_idx]
            mat_roughness = sphere_metal_fuzz[obj_idx] if smat == 1 else 1.0
            mat_metalness = 1.0 if smat == 1 else 0.0
            mat_trans     = 1.0 if smat == 2 else 0.0
            mat_emission  = mat_colour * sphere_emission[obj_idx] if smat == 3 else tm.vec3(0.0)

        # ── Emissive — MIS-weighted BRDF contribution ────────────────────────
        if mat_emission.max() > 0.0:
            w_brdf = 1.0
            # MIS only applies when the previous bounce was diffuse AND this emitter has an NEE counterpart.
            # Camera rays and specular/glass bounces always get full weight (no competing NEE strategy).
            if not prev_specular and hit_type == 2:
                lpdf = sphere_light_pdf(obj_idx, prev_hit_point, ray_d)
                w_brdf = power_heuristic(prev_brdf_pdf, lpdf)
            colour += throughput * w_brdf * mat_emission
            break

        # ── Per-bounce Sobol dimension base ──────────────────────────────────
        base = DIM_BOUNCE_BASE + bounce * DIMS_PER_BOUNCE

        # ── Direct light sampling (NEE) — diffuse/metal surfaces ─────────────
        is_diffuse = mat_metalness < 0.5 and mat_trans < 0.5
        if is_diffuse:
            diffuse_brdf = mat_colour / tm.pi
            lu1 = sobol(sample_idx, base + DIM_LIGHT_U, pixel_seed)
            lu2 = sobol(sample_idx, base + DIM_LIGHT_V, pixel_seed)
            for si in range(n_spheres):
                if sphere_material[si] == 3:
                    light_dir, lpdf = sphere_light_sample(si, hit_point, lu1, lu2)
                    if lpdf > 0.0:
                        cos_t = tm.dot(light_dir, normal)
                        if cos_t > 0.0:
                            t_max = (sphere_center[si] - hit_point).norm() + sphere_radius[si]
                            sh_t, sh_type, sh_idx, _, _ = scene_hit(hit_point, light_dir, 0.001, t_max)
                            if sh_type == 2 and sh_idx == si:
                                light_emission = sphere_colour[si] * sphere_emission[si]
                                brdf_pdf_for_light_dir = cos_t / tm.pi
                                w_light = power_heuristic(lpdf, brdf_pdf_for_light_dir)
                                colour += throughput * w_light * diffuse_brdf * cos_t * light_emission / lpdf

        # ── BRDF scatter ──────────────────────────────────────────────────────
        r_mat = sobol(sample_idx, base + DIM_MAT,       pixel_seed)
        su1   = sobol(sample_idx, base + DIM_SCATTER_U, pixel_seed)
        su2   = sobol(sample_idx, base + DIM_SCATTER_V, pixel_seed)

        scatter_dir, attenuation, brdf_pdf, did_scatter = scatter(
            mat_colour, mat_roughness, mat_metalness, mat_trans, ray_d, normal,
            r_mat, su1, su2
        )

        if not did_scatter:
            break

        throughput *= attenuation
        prev_hit_point = hit_point
        prev_specular  = mat_metalness >= 0.5 or mat_trans >= 0.5
        prev_brdf_pdf  = brdf_pdf
        ray_o = hit_point
        ray_d = normalize(scatter_dir)
    return colour


@ti.kernel
def render():
    img_height = Config.img_height
    img_width  = Config.img_width
    viewport_height = Config.viewport_height
    viewport_width = Config.viewport_width
    aperture   = Config.aperture
    focus_dist = Config.focus_dist
    antialising_samples = Config.antialising_samples
    camera_center = tm.vec3(Config.camera_center)

    # forward axis — direction the camera looks
    w = normalize(tm.vec3(Config.lookfrom - Config.lookat))

    # right axis — perpendicular to forward and up
    u = normalize(tm.cross(tm.vec3(Config.vup), w))

    # up axis — perpendicular to both (true up relative to camera)
    v = tm.cross(w, u)

    # how much to step in 3D space to move one pixel
    pixel_delta_u = u * (viewport_width / img_width)     # right
    pixel_delta_v = -v * (viewport_height / img_height)  # down

    # find the 3D position of the top-left corner of the viewport
    # start at camera, go forward (focal_length in -Z), then go left and up by half the viewport
    viewport_upper_left = (
        camera_center
        - Config.focal_length * w
        - (viewport_width / 2) * u
        + (viewport_height / 2) * v
    )

    # pixel00_loc is the center of the top-left pixel (not the corner of the viewport)
    # we offset by half a pixel in both directions to center within the pixel
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

    sqrt_spp = int(tm.sqrt(antialising_samples))
    use_stratified = sqrt_spp * sqrt_spp == antialising_samples

    for j, i in ti.ndrange(img_height, img_width):
        pseed = pixel_seeds[j, i]

        for s in range(antialising_samples):
            sample_du = 0.0
            sample_dv = 0.0
            if use_stratified:
                grid_c = s % sqrt_spp
                grid_r = s // sqrt_spp
                # Sobol replaces ti.random() inside each stratum cell
                sample_du = (grid_c + sobol(s, DIM_BOUNCE_BASE + DIM_SCATTER_U, pseed)) / sqrt_spp - 0.5
                sample_dv = (grid_r + sobol(s, DIM_BOUNCE_BASE + DIM_SCATTER_V, pseed)) / sqrt_spp - 0.5
            else:
                sample_du = sobol(s, DIM_BOUNCE_BASE + DIM_SCATTER_U, pseed) - 0.5
                sample_dv = sobol(s, DIM_BOUNCE_BASE + DIM_SCATTER_V, pseed) - 0.5

            du = i + sample_du
            dv = j + sample_dv

            # find the 3D center of this pixel on the viewport
            pixel_center = pixel00_loc + du * pixel_delta_u + dv * pixel_delta_v

            # DoF — Sobol replaces the two ti.random() calls
            dof_u  = sobol(s, DIM_DOF_U, pseed)
            dof_v  = sobol(s, DIM_DOF_V, pseed)
            angle  = dof_u * 2.0 * tm.pi
            radius = tm.sqrt(dof_v)

            offset = aperture * radius * (tm.cos(angle) * u + tm.sin(angle) * v)  # random point in aperture disk

            ray_o = camera_center + offset

            # the ray direction is from the camera toward that pixel's 3D position
            # this is what makes each pixel look in a slightly different direction
            pixel_direction = normalize(pixel_center - camera_center)
            focal_point = camera_center + focus_dist * pixel_direction  # fixed point in space
            ray_d = normalize(focal_point - ray_o)  # ray from lens point to focal point

            colour = ray_colour(ray_o, ray_d, s, pseed)

            framebuffer[j, i] += colour  # accumulate the colour for anti-aliasing

        framebuffer[j, i] /= antialising_samples  # average the samples for anti-aliasingoint to focal point

            colour = ray_colour(ray_o, ray_d)

            framebuffer[j, i] += colour  # accumulate the colour for anti-aliasing

        framebuffer[j, i] /= antialising_samples  # average the samples for anti-aliasing


def _collect_obj_tris(obj):
    """Collect triangles from a single mesh object's BVH."""
    tris = []
    stack = [obj.bvh]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            tris.extend(node.triangles)
        elif node.children is not None:
            stack.extend(node.children)
        else:
            stack.append(node.left)
            stack.append(node.right)
    return tris


def gather_triangles(scene):
    """Walk the scene and collect all Triangle objects into a flat list."""
    tris = []
    for obj in scene.objects:
        if hasattr(obj, 'bvh'):
            tris.extend(_collect_obj_tris(obj))
    return tris


def _build_flat_bvh(v0_arr, v1_arr, v2_arr, leaf_size=8):
    """
    Build a BVH from scratch using flat numpy arrays.
    Returns arrays ready to upload to GPU:
      bbox_min, bbox_max : (N, 3) float32
      left, right        : (N,)   int32   — child node indices, -1 for leaves
      tri_start, tri_end : (N,)   int32   — range into ordered_indices (leaves only)
      ordered_indices    : (M,)   int32   — flat triangle index list
    """
    records = []   # filled by build(); each entry is a tuple
    ordered = []   # flat list of triangle indices in leaf order

    def build(indices):
        node_idx = len(records)
        records.append(None)  # reserve slot, filled after children are known

        v0 = v0_arr[indices]
        v1 = v1_arr[indices]
        v2 = v2_arr[indices]
        bmin = np.minimum(np.minimum(v0, v1), v2).min(axis=0).astype(np.float32) - 1e-4
        bmax = np.maximum(np.maximum(v0, v1), v2).max(axis=0).astype(np.float32) + 1e-4

        if len(indices) <= leaf_size:
            # Leaf: store triangle indices in ordered list
            start = len(ordered)
            ordered.extend(indices.tolist())
            records[node_idx] = (bmin, bmax, -1, -1, start, len(ordered))
            return node_idx

        # Split on longest axis at median centroid
        centroids = (v0 + v1 + v2) / 3.0
        axis = int(np.argmax(centroids.max(axis=0) - centroids.min(axis=0)))
        order = np.argsort(centroids[:, axis])
        sorted_indices = indices[order]
        mid = len(sorted_indices) // 2

        left_idx  = build(sorted_indices[:mid])
        right_idx = build(sorted_indices[mid:])
        records[node_idx] = (bmin, bmax, left_idx, right_idx, 0, 0)
        return node_idx

    build(np.arange(len(v0_arr), dtype=np.int32))

    # Convert list of tuples to flat numpy arrays
    n = len(records)
    bbox_min_arr  = np.zeros((n, 3), dtype=np.float32)
    bbox_max_arr  = np.zeros((n, 3), dtype=np.float32)
    left_arr      = np.full(n, -1, dtype=np.int32)
    right_arr     = np.full(n, -1, dtype=np.int32)
    tri_start_arr = np.zeros(n, dtype=np.int32)
    tri_end_arr   = np.zeros(n, dtype=np.int32)

    for i, (bmin, bmax, l, r, ts, te) in enumerate(records):
        bbox_min_arr[i]  = bmin
        bbox_max_arr[i]  = bmax
        left_arr[i]      = l
        right_arr[i]     = r
        tri_start_arr[i] = ts
        tri_end_arr[i]   = te

    return bbox_min_arr, bbox_max_arr, left_arr, right_arr, tri_start_arr, tri_end_arr, np.array(ordered, dtype=np.int32)


def build(scene):
    global tri_v0, tri_v1, tri_v2, tri_colour
    global tri_roughness, tri_metalness, tri_transmission, tri_emission
    global tri_normal, tri_n0, tri_n1, tri_n2, tri_has_smooth
    global tri_uv0, tri_uv1, tri_uv2, tri_tex_id
    global tex_atlas, n_textures
    global bvh_bbox_min, bvh_bbox_max, bvh_left, bvh_right
    global bvh_tri_start, bvh_tri_end, bvh_tri_indices
    global plane_point, plane_normal, plane_colour, plane_material
    global plane_metal_fuzz, plane_refraction_index, plane_emission, n_planes
    global sphere_center, sphere_radius, sphere_colour, sphere_material
    global sphere_metal_fuzz, sphere_refraction_index, sphere_emission, n_spheres

    all_tris = gather_triangles(scene)
    n = len(all_tris)
    print(f"[Build] {n:,} triangles — uploading to GPU...")

    materials = {None: 0, "metal": 1, "glass": 2, "emissive": 3, "absorbing": 4}

    # ── Extract triangle data into numpy arrays ──────────────────────────────
    v0_np  = np.array([t.v0  for t in all_tris], dtype=np.float32)
    v1_np  = np.array([t.v1  for t in all_tris], dtype=np.float32)
    v2_np  = np.array([t.v2  for t in all_tris], dtype=np.float32)

    colour_np       = np.array([t.colour.astype(np.float32)   for t in all_tris], dtype=np.float32)
    roughness_np    = np.array([t.roughness                    for t in all_tris], dtype=np.float32)
    metalness_np    = np.array([t.metalness                    for t in all_tris], dtype=np.float32)
    transmission_np = np.array([t.transmission                 for t in all_tris], dtype=np.float32)
    emission_np     = np.array([t.emission.astype(np.float32)  for t in all_tris], dtype=np.float32)
    normal_np    = np.array([t._normal.astype(np.float32)                      for t in all_tris], dtype=np.float32)
    smooth_np    = np.array([1 if t.n0 is not None else 0                      for t in all_tris], dtype=np.int32)
    n0_np = np.array([t.n0 if t.n0 is not None else t._normal for t in all_tris], dtype=np.float32)
    n1_np = np.array([t.n1 if t.n1 is not None else t._normal for t in all_tris], dtype=np.float32)
    n2_np = np.array([t.n2 if t.n2 is not None else t._normal for t in all_tris], dtype=np.float32)

    _z2 = np.zeros(2, dtype=np.float32)
    uv0_np    = np.array([t.uv0    if hasattr(t, 'uv0') else _z2 for t in all_tris], dtype=np.float32)
    uv1_np    = np.array([t.uv1    if hasattr(t, 'uv1') else _z2 for t in all_tris], dtype=np.float32)
    uv2_np    = np.array([t.uv2    if hasattr(t, 'uv2') else _z2 for t in all_tris], dtype=np.float32)
    tex_id_np = np.array([t.tex_id if hasattr(t, 'tex_id') else -1 for t in all_tris], dtype=np.int32)

    # ── Collect textures; remap per-mesh local tex_ids to global atlas ───────
    all_textures = []
    for obj in scene.objects:
        if hasattr(obj, 'textures') and obj.textures:
            all_textures.extend(obj.textures)
    n_textures = len(all_textures)

    # Rebuild tex_id_np with global offsets by tracking which object each tri belongs to
    global_offset = 0
    tex_id_np_remapped = tex_id_np.copy()
    i = 0
    for obj in scene.objects:
        if not hasattr(obj, 'bvh'):
            global_offset += len(getattr(obj, 'textures', []))
            continue
        obj_tris = _collect_obj_tris(obj)
        n_obj = len(obj_tris)
        obj_offset = global_offset
        if obj_offset > 0:
            mask = tex_id_np_remapped[i:i + n_obj] >= 0
            tex_id_np_remapped[i:i + n_obj][mask] += obj_offset
        i += n_obj
        global_offset += len(getattr(obj, 'textures', []))
    tex_id_np = tex_id_np_remapped

    # ── Build flat BVH ───────────────────────────────────────────────────────
    print(f"[Build] Building BVH...")
    bmin_np, bmax_np, left_np, right_np, ts_np, te_np, tidx_np = _build_flat_bvh(v0_np, v1_np, v2_np)
    n_nodes = len(bmin_np)
    print(f"[Build] BVH: {n_nodes:,} nodes")

    # ── Allocate triangle fields ─────────────────────────────────────────────
    tri_v0               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_v1               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_v2               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_colour           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_roughness        = ti.field(dtype=ti.f32, shape=n)
    tri_metalness        = ti.field(dtype=ti.f32, shape=n)
    tri_transmission     = ti.field(dtype=ti.f32, shape=n)
    tri_emission         = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_normal           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n0               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n1               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n2               = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_has_smooth       = ti.field(dtype=ti.i32, shape=n)
    tri_uv0              = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_uv1              = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_uv2              = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_tex_id           = ti.field(dtype=ti.i32, shape=n)

    # ── Allocate texture atlas ───────────────────────────────────────────────
    atlas_count = max(n_textures, 1)
    tex_atlas = ti.Vector.field(3, dtype=ti.f32, shape=(atlas_count, TEX_SIZE, TEX_SIZE))
    if n_textures > 0:
        atlas_np = np.zeros((n_textures, TEX_SIZE, TEX_SIZE, 3), dtype=np.float32)
        for ti_idx, img in enumerate(all_textures):
            resized = cv2.resize(img, (TEX_SIZE, TEX_SIZE), interpolation=cv2.INTER_LINEAR)
            atlas_np[ti_idx] = resized
        tex_atlas.from_numpy(atlas_np)

    # ── Allocate BVH fields ──────────────────────────────────────────────────
    bvh_bbox_min    = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    bvh_bbox_max    = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    bvh_left        = ti.field(dtype=ti.i32, shape=n_nodes)
    bvh_right       = ti.field(dtype=ti.i32, shape=n_nodes)
    bvh_tri_start   = ti.field(dtype=ti.i32, shape=n_nodes)
    bvh_tri_end     = ti.field(dtype=ti.i32, shape=n_nodes)
    bvh_tri_indices = ti.field(dtype=ti.i32, shape=len(tidx_np))

    tri_v0.from_numpy(v0_np)
    tri_v1.from_numpy(v1_np)
    tri_v2.from_numpy(v2_np)
    tri_colour.from_numpy(colour_np)
    tri_roughness.from_numpy(roughness_np)
    tri_metalness.from_numpy(metalness_np)
    tri_transmission.from_numpy(transmission_np)
    tri_emission.from_numpy(emission_np)
    tri_normal.from_numpy(normal_np)
    tri_has_smooth.from_numpy(smooth_np)
    tri_n0.from_numpy(n0_np)
    tri_n1.from_numpy(n1_np)
    tri_n2.from_numpy(n2_np)
    tri_uv0.from_numpy(uv0_np)
    tri_uv1.from_numpy(uv1_np)
    tri_uv2.from_numpy(uv2_np)
    tri_tex_id.from_numpy(tex_id_np)

    # ── Fill BVH fields (bulk upload via numpy) ──────────────────────────────
    bvh_bbox_min.from_numpy(bmin_np)
    bvh_bbox_max.from_numpy(bmax_np)
    bvh_left.from_numpy(left_np)
    bvh_right.from_numpy(right_np)
    bvh_tri_start.from_numpy(ts_np)
    bvh_tri_end.from_numpy(te_np)
    bvh_tri_indices.from_numpy(tidx_np)

    # ── Planes ───────────────────────────────────────────────────────────────
    from objects import Plane, Sphere
    planes  = [o for o in scene.objects if isinstance(o, Plane)]
    spheres = [o for o in scene.objects if isinstance(o, Sphere)]
    n_planes  = len(planes)
    n_spheres = len(spheres)

    if n_planes > 0:
        plane_point    = ti.Vector.field(3, dtype=ti.f32, shape=n_planes)
        plane_normal   = ti.Vector.field(3, dtype=ti.f32, shape=n_planes)
        plane_colour   = ti.Vector.field(3, dtype=ti.f32, shape=n_planes)
        plane_material = ti.field(dtype=ti.i32, shape=n_planes)
        plane_metal_fuzz       = ti.field(dtype=ti.f32, shape=n_planes)
        plane_refraction_index = ti.field(dtype=ti.f32, shape=n_planes)
        plane_emission         = ti.field(dtype=ti.f32, shape=n_planes)
        for i, p in enumerate(planes):
            plane_point[i]    = p.center.astype(np.float32)
            plane_normal[i]   = p.normal.astype(np.float32)
            plane_colour[i]   = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # checkerboard handled in shader
            plane_material[i] = materials.get(p.material, 0)
            plane_metal_fuzz[i]       = 0.0
            plane_refraction_index[i] = 1.5
            plane_emission[i]         = 0.0
    else:
        # Allocate dummy 1-element fields so the kernel compiles
        plane_point    = ti.Vector.field(3, dtype=ti.f32, shape=1)
        plane_normal   = ti.Vector.field(3, dtype=ti.f32, shape=1)
        plane_colour   = ti.Vector.field(3, dtype=ti.f32, shape=1)
        plane_material = ti.field(dtype=ti.i32, shape=1)
        plane_metal_fuzz       = ti.field(dtype=ti.f32, shape=1)
        plane_refraction_index = ti.field(dtype=ti.f32, shape=1)
        plane_emission         = ti.field(dtype=ti.f32, shape=1)

    if n_spheres > 0:
        sphere_center   = ti.Vector.field(3, dtype=ti.f32, shape=n_spheres)
        sphere_radius   = ti.field(dtype=ti.f32, shape=n_spheres)
        sphere_colour   = ti.Vector.field(3, dtype=ti.f32, shape=n_spheres)
        sphere_material = ti.field(dtype=ti.i32, shape=n_spheres)
        sphere_metal_fuzz       = ti.field(dtype=ti.f32, shape=n_spheres)
        sphere_refraction_index = ti.field(dtype=ti.f32, shape=n_spheres)
        sphere_emission         = ti.field(dtype=ti.f32, shape=n_spheres)
        for i, s in enumerate(spheres):
            sphere_center[i]   = s.center.astype(np.float32)
            sphere_radius[i]   = float(s.radius)
            sphere_colour[i]   = s.colour.astype(np.float32)
            sphere_material[i] = materials.get(s.material, 0)
            sphere_metal_fuzz[i]       = float(s.metal_fuzz)
            sphere_refraction_index[i] = float(s.refraction_index)
            sphere_emission[i]         = float(s.emission_intensity)
    else:
        sphere_center   = ti.Vector.field(3, dtype=ti.f32, shape=1)
        sphere_radius   = ti.field(dtype=ti.f32, shape=1)
        sphere_colour   = ti.Vector.field(3, dtype=ti.f32, shape=1)
        sphere_material = ti.field(dtype=ti.i32, shape=1)
        sphere_metal_fuzz       = ti.field(dtype=ti.f32, shape=1)
        sphere_refraction_index = ti.field(dtype=ti.f32, shape=1)
        sphere_emission         = ti.field(dtype=ti.f32, shape=1)

    print(f"[Build] {n_planes} plane(s), {n_spheres} sphere(s) uploaded.")
    print(f"[Build] Done.")


def save_scene(path):
    """Save all GPU-ready numpy arrays to a compressed .npz file."""
    print(f"[Scene] Saving to {path}...")
    np.savez_compressed(
        path,
        # triangles
        tri_v0           = tri_v0.to_numpy(),
        tri_v1           = tri_v1.to_numpy(),
        tri_v2           = tri_v2.to_numpy(),
        tri_colour       = tri_colour.to_numpy(),
        tri_roughness    = tri_roughness.to_numpy(),
        tri_metalness    = tri_metalness.to_numpy(),
        tri_transmission = tri_transmission.to_numpy(),
        tri_emission     = tri_emission.to_numpy(),
        tri_normal       = tri_normal.to_numpy(),
        tri_n0           = tri_n0.to_numpy(),
        tri_n1           = tri_n1.to_numpy(),
        tri_n2           = tri_n2.to_numpy(),
        tri_has_smooth   = tri_has_smooth.to_numpy(),
        tri_uv0          = tri_uv0.to_numpy(),
        tri_uv1          = tri_uv1.to_numpy(),
        tri_uv2          = tri_uv2.to_numpy(),
        tri_tex_id       = tri_tex_id.to_numpy(),
        n_textures       = np.array(n_textures),
        tex_atlas        = tex_atlas.to_numpy(),
        # BVH
        bvh_bbox_min    = bvh_bbox_min.to_numpy(),
        bvh_bbox_max    = bvh_bbox_max.to_numpy(),
        bvh_left        = bvh_left.to_numpy(),
        bvh_right       = bvh_right.to_numpy(),
        bvh_tri_start   = bvh_tri_start.to_numpy(),
        bvh_tri_end     = bvh_tri_end.to_numpy(),
        bvh_tri_indices = bvh_tri_indices.to_numpy(),
        # planes
        n_planes         = np.array(n_planes),
        plane_point      = plane_point.to_numpy(),
        plane_normal     = plane_normal.to_numpy(),
        plane_colour     = plane_colour.to_numpy(),
        plane_material   = plane_material.to_numpy(),
        plane_metal_fuzz = plane_metal_fuzz.to_numpy(),
        plane_refraction_index = plane_refraction_index.to_numpy(),
        plane_emission   = plane_emission.to_numpy(),
        # spheres
        n_spheres        = np.array(n_spheres),
        sphere_center    = sphere_center.to_numpy(),
        sphere_radius    = sphere_radius.to_numpy(),
        sphere_colour    = sphere_colour.to_numpy(),
        sphere_material  = sphere_material.to_numpy(),
        sphere_metal_fuzz       = sphere_metal_fuzz.to_numpy(),
        sphere_refraction_index = sphere_refraction_index.to_numpy(),
        sphere_emission  = sphere_emission.to_numpy(),
    )
    print(f"[Scene] Saved.")


def load_scene(path):
    """Load a saved scene from .npz and upload directly to GPU, skipping BVH build."""
    global tri_v0, tri_v1, tri_v2, tri_colour
    global tri_roughness, tri_metalness, tri_transmission, tri_emission
    global tri_normal, tri_n0, tri_n1, tri_n2, tri_has_smooth
    global tri_uv0, tri_uv1, tri_uv2, tri_tex_id
    global tex_atlas, n_textures
    global bvh_bbox_min, bvh_bbox_max, bvh_left, bvh_right
    global bvh_tri_start, bvh_tri_end, bvh_tri_indices
    global plane_point, plane_normal, plane_colour, plane_material
    global plane_metal_fuzz, plane_refraction_index, plane_emission, n_planes
    global sphere_center, sphere_radius, sphere_colour, sphere_material
    global sphere_metal_fuzz, sphere_refraction_index, sphere_emission, n_spheres

    print(f"[Scene] Loading from {path}...")
    d = np.load(path)

    n  = len(d['tri_v0'])
    tri_v0           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_v1           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_v2           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_colour       = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_roughness    = ti.field(dtype=ti.f32, shape=n)
    tri_metalness    = ti.field(dtype=ti.f32, shape=n)
    tri_transmission = ti.field(dtype=ti.f32, shape=n)
    tri_emission     = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_normal       = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n0           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n1           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_n2           = ti.Vector.field(3, dtype=ti.f32, shape=n)
    tri_has_smooth   = ti.field(dtype=ti.i32, shape=n)
    tri_uv0          = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_uv1          = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_uv2          = ti.Vector.field(2, dtype=ti.f32, shape=n)
    tri_tex_id       = ti.field(dtype=ti.i32, shape=n)

    tri_v0.from_numpy(d['tri_v0'])
    tri_v1.from_numpy(d['tri_v1'])
    tri_v2.from_numpy(d['tri_v2'])
    tri_colour.from_numpy(d['tri_colour'])
    tri_roughness.from_numpy(d['tri_roughness'])
    tri_metalness.from_numpy(d['tri_metalness'])
    tri_transmission.from_numpy(d['tri_transmission'])
    tri_emission.from_numpy(d['tri_emission'])
    tri_normal.from_numpy(d['tri_normal'])
    tri_n0.from_numpy(d['tri_n0'])
    tri_n1.from_numpy(d['tri_n1'])
    tri_n2.from_numpy(d['tri_n2'])
    tri_has_smooth.from_numpy(d['tri_has_smooth'])
    tri_uv0.from_numpy(d['tri_uv0'])
    tri_uv1.from_numpy(d['tri_uv1'])
    tri_uv2.from_numpy(d['tri_uv2'])
    tri_tex_id.from_numpy(d['tri_tex_id'])

    n_textures = int(d['n_textures'])
    atlas_data = d['tex_atlas']
    atlas_count = max(n_textures, 1)
    tex_atlas = ti.Vector.field(3, dtype=ti.f32, shape=(atlas_count, TEX_SIZE, TEX_SIZE))
    tex_atlas.from_numpy(atlas_data)

    nn = len(d['bvh_bbox_min'])
    bvh_bbox_min    = ti.Vector.field(3, dtype=ti.f32, shape=nn)
    bvh_bbox_max    = ti.Vector.field(3, dtype=ti.f32, shape=nn)
    bvh_left        = ti.field(dtype=ti.i32, shape=nn)
    bvh_right       = ti.field(dtype=ti.i32, shape=nn)
    bvh_tri_start   = ti.field(dtype=ti.i32, shape=nn)
    bvh_tri_end     = ti.field(dtype=ti.i32, shape=nn)
    bvh_tri_indices = ti.field(dtype=ti.i32, shape=len(d['bvh_tri_indices']))

    bvh_bbox_min.from_numpy(d['bvh_bbox_min'])
    bvh_bbox_max.from_numpy(d['bvh_bbox_max'])
    bvh_left.from_numpy(d['bvh_left'])
    bvh_right.from_numpy(d['bvh_right'])
    bvh_tri_start.from_numpy(d['bvh_tri_start'])
    bvh_tri_end.from_numpy(d['bvh_tri_end'])
    bvh_tri_indices.from_numpy(d['bvh_tri_indices'])

    n_planes = int(d['n_planes'])
    ns = max(n_planes, 1)
    plane_point            = ti.Vector.field(3, dtype=ti.f32, shape=ns)
    plane_normal           = ti.Vector.field(3, dtype=ti.f32, shape=ns)
    plane_colour           = ti.Vector.field(3, dtype=ti.f32, shape=ns)
    plane_material         = ti.field(dtype=ti.i32, shape=ns)
    plane_metal_fuzz       = ti.field(dtype=ti.f32, shape=ns)
    plane_refraction_index = ti.field(dtype=ti.f32, shape=ns)
    plane_emission         = ti.field(dtype=ti.f32, shape=ns)
    plane_point.from_numpy(d['plane_point'])
    plane_normal.from_numpy(d['plane_normal'])
    plane_colour.from_numpy(d['plane_colour'])
    plane_material.from_numpy(d['plane_material'])
    plane_metal_fuzz.from_numpy(d['plane_metal_fuzz'])
    plane_refraction_index.from_numpy(d['plane_refraction_index'])
    plane_emission.from_numpy(d['plane_emission'])

    n_spheres = int(d['n_spheres'])
    ns = max(n_spheres, 1)
    sphere_center           = ti.Vector.field(3, dtype=ti.f32, shape=ns)
    sphere_radius           = ti.field(dtype=ti.f32, shape=ns)
    sphere_colour           = ti.Vector.field(3, dtype=ti.f32, shape=ns)
    sphere_material         = ti.field(dtype=ti.i32, shape=ns)
    sphere_metal_fuzz       = ti.field(dtype=ti.f32, shape=ns)
    sphere_refraction_index = ti.field(dtype=ti.f32, shape=ns)
    sphere_emission         = ti.field(dtype=ti.f32, shape=ns)
    sphere_center.from_numpy(d['sphere_center'])
    sphere_radius.from_numpy(d['sphere_radius'])
    sphere_colour.from_numpy(d['sphere_colour'])
    sphere_material.from_numpy(d['sphere_material'])
    sphere_metal_fuzz.from_numpy(d['sphere_metal_fuzz'])
    sphere_refraction_index.from_numpy(d['sphere_refraction_index'])
    sphere_emission.from_numpy(d['sphere_emission'])

    print(f"[Scene] Loaded {n:,} triangles, {nn:,} BVH nodes, "
          f"{n_planes} plane(s), {n_spheres} sphere(s).")


@ti.kernel
def _clear():
    for j, i in ti.ndrange(Config.img_height, Config.img_width):
        framebuffer[j, i] = tm.vec3(0.0)


def run():
    """Clear framebuffer, run the render kernel, save output."""
    import os, time
    os.makedirs("img", exist_ok=True)
    os.makedirs("hdr", exist_ok=True)

    print(f"[Render] Rendering {Config.img_width}x{Config.img_height}, {Config.antialising_samples} spp...")
    _clear()
    t0 = time.time()
    render()
    ti.sync()
    elapsed = time.time() - t0
    print(f"[Render] Done in {elapsed:.1f}s")

    img = framebuffer.to_numpy()                    # (H, W, 3) float32 — raw HDR
    np.save(Config.hdr_save_path, img)
    img_mapped = _tone_map_np(img)                  # HDR → [0, 1]
    out = apply_effect(img_mapped)                  # bloom on tone-mapped image
    cv2.imwrite(Config.save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[Render] Saved {Config.save_path}")


def _tone_map_np(hdr: np.ndarray) -> np.ndarray:
    """Apply filmic (ACES approximation) tone mapping to a float32 HDR image."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0)