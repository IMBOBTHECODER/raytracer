import pyassimp
from config import Config
import numpy as np
import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

tri_v0              = None
tri_v1              = None
tri_v2              = None
tri_colour          = None
tri_material        = None
tri_metal_fuzz      = None
tri_refraction_index= None
tri_emission        = None
tri_normal          = None
tri_n0              = None
tri_n1              = None
tri_n2              = None
tri_has_smooth      = None
tri_uv0             = None
tri_uv1             = None
tri_uv2             = None
tri_tex_id          = None

tex_atlas  = None
n_textures = 0
TEX_SIZE   = 512

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
    return tm.vec3(0.0, 0.0, 0.0)


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
    # Warm orange-gold tint on bloom (cinematic key-light feel)
    blur_hdr   *= np.array([1.2, 0.95, 0.6], dtype=np.float32)
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
def get_colour(tri_idx):
    result = tm.vec3(0.0)
    if tri_material[tri_idx] == 3:  # emissive
        result = tri_colour[tri_idx] * tri_emission[tri_idx]
    return result

@ti.func
def random_unit_vec():
    # Random point on unit sphere via rejection — keep sampling until inside unit sphere
    v = tm.vec3(0.0)
    while True:
        v = tm.vec3(ti.random() * 2 - 1, ti.random() * 2 - 1, ti.random() * 2 - 1)
        if v.norm_sqr() <= 1.0:
            break
    return normalize(v)

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
def sphere_light_sample(sphere_idx, hit_point):
    """Sample a direction toward sphere light. Returns (dir, pdf); pdf=0 on degenerate."""
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

        r1 = ti.random()
        r2 = ti.random()
        cos_t = 1.0 - r1 * (1.0 - cos_max)
        sin_t = tm.sqrt(tm.max(0.0, 1.0 - cos_t * cos_t))
        phi   = 2.0 * tm.pi * r2
        light_dir = normalize(sin_t * tm.cos(phi) * uu + sin_t * tm.sin(phi) * vv + cos_t * w)

    return light_dir, pdf


@ti.func
def scatter(colour, roughness, metalness, transmission, ray_d, normal):
    """Material BRDF — takes raw material values, works for any object type."""
    scatter_dir = tm.vec3(0.0)
    attenuation = colour
    did_scatter = False

    r = ti.random()

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
        if eta * sin_theta > 1.0 or ti.random() < reflectance:
            scatter_dir = d - 2 * tm.dot(d, n) * n
        else:
            d_perp     = eta * (d + cos_theta * n)
            d_parallel = -tm.sqrt(ti.abs(1.0 - d_perp.norm_sqr())) * n
            scatter_dir = d_perp + d_parallel
        attenuation = colour
        did_scatter = True

    elif r < transmission + metalness:  # Metal
        d = normalize(ray_d)
        reflected = d - 2 * tm.dot(d, normal) * normal
        reflected += roughness * random_unit_vec()
        did_scatter = tm.dot(reflected, normal) > 0
        scatter_dir = reflected

    else:  # Diffuse
        rand_vec = random_unit_vec()
        if tm.dot(rand_vec, normal) < 0:
            rand_vec = -rand_vec
        scatter_direction = normal + rand_vec
        if scatter_direction.norm() < 1e-8:
            scatter_direction = normal
        scatter_dir = scatter_direction
        did_scatter = True

    return scatter_dir, attenuation, did_scatter


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
def ray_colour(ray_o, ray_d):
    colour     = tm.vec3(0.0)
    throughput = tm.vec3(1.0)
    nee_done   = False  # skip emitter emission after diffuse bounces (NEE already sampled them)

    for depth in range(Config.depth_limit):
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

        # ── Emissive — add and terminate ──────────────────────────────────────
        if mat_emission.max() > 0.0:
            if not nee_done:
                colour += throughput * mat_emission
            break

        # ── Direct light sampling (NEE) — diffuse/metal surfaces ─────────────
        is_diffuse = mat_metalness < 0.5 and mat_trans < 0.5
        if is_diffuse:
            diffuse_brdf = mat_colour / tm.pi
            for si in range(n_spheres):
                if sphere_material[si] == 3:
                    light_dir, lpdf = sphere_light_sample(si, hit_point)
                    if lpdf > 0.0:
                        cos_t = tm.dot(light_dir, normal)
                        if cos_t > 0.0:
                            t_max = (sphere_center[si] - hit_point).norm() + sphere_radius[si]
                            sh_t, sh_type, sh_idx, _, _ = scene_hit(hit_point, light_dir, 0.001, t_max)
                            if sh_type == 2 and sh_idx == si:
                                colour += throughput * diffuse_brdf * cos_t \
                                          * sphere_colour[si] * sphere_emission[si] / lpdf

        # ── BRDF scatter ──────────────────────────────────────────────────────
        scatter_dir, attenuation, did_scatter = scatter(
            mat_colour, mat_roughness, mat_metalness, mat_trans, ray_d, normal
        )

        if not did_scatter:
            break

        throughput *= attenuation
        nee_done = is_diffuse
        ray_o    = hit_point
        ray_d    = normalize(scatter_dir)

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
        for s in range(antialising_samples):
            sample_du = 0.0
            sample_dv = 0.0
            if use_stratified:
                grid_c = s % sqrt_spp
                grid_r = s // sqrt_spp
                sample_du = (grid_c + ti.random()) / sqrt_spp - 0.5
                sample_dv = (grid_r + ti.random()) / sqrt_spp - 0.5
            else:
                sample_du = ti.random() - 0.5
                sample_dv = ti.random() - 0.5

            du = i + sample_du
            dv = j + sample_dv

            # find the 3D center of this pixel on the viewport
            pixel_center = pixel00_loc + du * pixel_delta_u + dv * pixel_delta_v

            angle  = ti.random() * 2.0 * tm.pi
            radius = tm.sqrt(ti.random())

            offset = aperture * radius * (tm.cos(angle) * u + tm.sin(angle) * v)  # random point in aperture disk

            ray_o = camera_center + offset

            # the ray direction is from the camera toward that pixel's 3D position
            # this is what makes each pixel look in a slightly different direction
            pixel_direction = normalize(pixel_center - camera_center)
            focal_point = camera_center + focus_dist * pixel_direction  # fixed point in space
            ray_d = normalize(focal_point - ray_o)  # ray from lens point to focal point

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