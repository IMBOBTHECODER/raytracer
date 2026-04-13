from functions import normalize, lerp
from config import Config
from mesh import _aabb_hit
import cv2
import numpy as np
import multiprocessing as mp
import threading
from numba import njit
import time


@njit(cache=True)
def background(direction: np.ndarray) -> np.ndarray:
    unit_direction = normalize(direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return lerp(np.array([0.9, 0.9, 0.9]), np.array([1, 0.945, 0.827]), t)

def tone_map(hdr):
    # Filmic (ACES approximation) — compresses highlights without clipping
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0, 1)


def apply_effect(linear: np.ndarray):
    """Apply bloom post-processing."""
    linear = linear.astype(np.float32)

    # Bloom — only very bright highlights bleed
    bright_hdr = np.clip(linear - 0.85, 0, None)
    blur_tight = cv2.GaussianBlur(bright_hdr, (51,  51),  0)
    blur_wide  = cv2.GaussianBlur(bright_hdr, (201, 201), 0)
    blur_hdr   = (blur_tight * 0.4 + blur_wide * 0.5) * 0.35
    # Cool/blue tint on the bloom glow
    blur_hdr  *= np.array([0.6, 0.85, 1.4], dtype=np.float32)
    bloomed = linear + blur_hdr

    out = (np.clip(bloomed, 0, 1) * 255).astype(np.uint8)
    return out


# ────────────────────────────────────────────────────────────────────────────

def ray_colour(t_min, t_max, ray_o, ray_d, world, depth=0):
    if depth >= Config.depth_limit:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)  # black for rays that exceed depth limit

    t, obj = world.hit(ray_o, ray_d, t_min, t_max)
    if obj is None:
        return background(ray_d)  # background colour if ray hits nothing

    hit_point = ray_o + t * ray_d
    normal = obj.get_normal(ray_d, t)

    scatter_direction = obj.reflect(ray_d, normal)

    if scatter_direction is None:
        return obj.object_colour(hit_point)  # emissive or absorbing both return here

    scattered_d = normalize(scatter_direction)

    return obj.object_colour(hit_point) * obj.colour_multiplier * ray_colour(t_min, t_max, hit_point, scattered_d, world, depth + 1)


def render_worker(j_start, i_start, antialising_samples, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, u, v):
    framebuffer = np.frombuffer(shared_array.get_obj()).reshape((Config.img_height, Config.img_width, 3))
    world = shared_world

    j_end = min(j_start + Config.tile_size, Config.img_height)
    i_end = min(i_start + Config.tile_size, Config.img_width)

    # Stratified sampling: if spp is a perfect square (4, 16, 64 ...),
    # divide each pixel into a sqrt_spp × sqrt_spp grid and place one sample
    # per cell with random jitter. Guarantees coverage, reduces clumping.
    sqrt_spp = int(np.sqrt(antialising_samples))
    use_stratified = sqrt_spp * sqrt_spp == antialising_samples
    if use_stratified:
        # Stratum row/col indices — same for every pixel, computed once
        grid_c = np.arange(antialising_samples) % sqrt_spp   # column: 0..sqrt_spp-1
        grid_r = np.arange(antialising_samples) // sqrt_spp  # row:    0..sqrt_spp-1

    # Aperture disk randoms — still fully random (not stratified)
    total = (j_end - j_start) * (i_end - i_start) * antialising_samples
    rng_angle  = np.random.uniform(0, 2 * np.pi, size=total)
    rng_radius = np.random.uniform(0, 1, size=total)
    idx = 0

    for j in range(j_start, j_end):
        for i in range(i_start, i_end):
            if use_stratified:
                # One jitter per stratum cell, scaled to [-0.5, 0.5] within the pixel
                jitter_u = np.random.uniform(0, 1, antialising_samples)
                jitter_v = np.random.uniform(0, 1, antialising_samples)
                sample_du = (grid_c + jitter_u) / sqrt_spp - 0.5
                sample_dv = (grid_r + jitter_v) / sqrt_spp - 0.5
            else:
                sample_du = np.random.uniform(-0.5, 0.5, antialising_samples)
                sample_dv = np.random.uniform(-0.5, 0.5, antialising_samples)

            for s in range(antialising_samples):
                du = i + sample_du[s]
                dv = j + sample_dv[s]

                # find the 3D center of this pixel on the viewport
                pixel_center = pixel00_loc + du * pixel_delta_u + dv * pixel_delta_v

                angle  = rng_angle[idx]
                radius = np.sqrt(rng_radius[idx])
                idx += 1

                offset = Config.aperture * radius * (np.cos(angle) * u + np.sin(angle) * v)  # random point in aperture disk

                ray_o = camera_center + offset

                # the ray direction is from the camera toward that pixel's 3D position
                # this is what makes each pixel look in a slightly different direction
                pixel_direction = normalize(pixel_center - camera_center)
                focal_point = camera_center + Config.focus_dist * pixel_direction  # fixed point in space
                ray_d = normalize(focal_point - ray_o)  # ray from lens point to focal point

                colour = ray_colour(0.008, float('inf'), ray_o, ray_d, world, depth=0)

                framebuffer[j, i] += colour  # accumulate the colour for anti-aliasing

            framebuffer[j, i] /= antialising_samples  # average the samples for anti-aliasing

def render_core(world=None, shared=None):
    # forward axis — direction the camera looks
    w = normalize(Config.lookfrom - Config.lookat)

    # right axis — perpendicular to forward and up
    u = normalize(np.cross(Config.vup, w))

    # up axis — perpendicular to both (true up relative to camera)
    v = np.cross(w, u)

    # how much to step in 3D space to move one pixel
    pixel_delta_u = u * (Config.viewport_width / Config.img_width)     # right
    pixel_delta_v = -v * (Config.viewport_height / Config.img_height)  # down

    # find the 3D position of the top-left corner of the viewport
    # start at camera, go forward (focal_length in -Z), then go left and up by half the viewport
    viewport_upper_left = (
        Config.camera_center
        - Config.focal_length * w
        - (Config.viewport_width / 2) * u
        + (Config.viewport_height / 2) * v
    )

    # pixel00_loc is the center of the top-left pixel (not the corner of the viewport)
    # we offset by half a pixel in both directions to center within the pixel
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

    tiles = [
        (j, i, Config.antialising_samples, pixel00_loc, pixel_delta_u, pixel_delta_v, Config.camera_center, u, v)
        for j in range(0, Config.img_height, Config.tile_size)
        for i in range(0, Config.img_width, Config.tile_size)
    ]
    print(f"[Render] Spawning {min(Config.num_workers, len(tiles))} workers for {len(tiles)} tiles ({Config.img_width}x{Config.img_height}, {Config.antialising_samples} spp)...")
    with mp.Pool(processes=min(Config.num_workers, len(tiles)), initializer=init_worker, initargs=(world, shared)) as pool:
        pool.starmap(render_worker, tiles, chunksize=Config.starmap_chunksize)
    print(f"[Render] All tiles done.")


shared_world = None
shared_array = None

def render(world):
    print(f"[Render] Initialising framebuffer ({Config.img_width}x{Config.img_height})...")
    shared = mp.Array('d', Config.img_height * Config.img_width * 3)
    framebuffer = np.frombuffer(shared.get_obj()).reshape((Config.img_height, Config.img_width, 3))

    # Preview loop runs in a thread so the Pool stays on the main thread (required on Windows)
    stop_preview = threading.Event()
    def preview_loop():
        while not stop_preview.is_set():
            # Quick gamma preview so you can see progress during render
            frame = (np.sqrt(np.clip(framebuffer, 0, 1)) * 255).astype(np.uint8)
            cv2.imwrite(Config.save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            time.sleep(1)

    thread = threading.Thread(target=preview_loop, daemon=True)
    thread.start()

    t0 = time.time()
    render_core(world, shared)

    stop_preview.set()
    thread.join()

    elapsed = time.time() - t0
    print(f"[Render] Render complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final HDR buffer and apply post-processing
    np.save(Config.hdr_save_path, framebuffer.copy())
    print(f"[Render] Applying post-processing...")
    out = apply_effect(framebuffer)
    cv2.imwrite(Config.save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[Render] Saved {Config.save_path}")

def init_worker(world, array):
    global shared_world, shared_array
    shared_world = world
    shared_array = array
    warmup()  # JIT compile the functions in the worker process

def warmup():
    # JIT warmup — compile all @njit functions before rendering begins
    _v = np.array([1.0, 2.0, 3.0])
    normalize(_v)
    lerp(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.7, 1.0]), 0.5)
    _aabb_hit(np.zeros(3), np.ones(3), np.zeros(3), np.array([1.0, 1.0, 1.0]), 0.0, 1e30)
    background(_v)
