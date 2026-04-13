import cv2
import numpy as np
import multiprocessing as mp
import threading
from numba import njit
import time

img_width = 512
img_height = 512
aspect_ratio = img_width / img_height

lookfrom = np.array([3.0, 1.5, 2.0])   # camera position
lookat = np.array([0.0, 0.5, -2.0])   # point camera is looking at
vup = np.array([0.0, 1.0, 0.0])       # "up" direction for the camera


# focal_length is the distance from the camera to the viewport
# larger = more zoomed in, smaller = more zoomed out
focal_length = 1.0

# the viewport is a rectangle in 3D space that we shoot rays through
# viewport_height=2.0 is arbitrary — it sets the scale of the scene
viewport_height = 2.0
viewport_width = viewport_height * aspect_ratio

# The camera poistion (currently the orgin)
camera_center = lookfrom

# 5-10 samples (development), 50-100 samples (final)
antialising_samples = 32 # number of rays to shoot per pixel for anti-aliasing

# 15 standard
depth_limit = 8 # maximum recursion depth for ray bounces (to prevent infinite recursion)

save_path = "img/render.png"
hdr_save_path = "hdr/render_final.npy"

aperture = 0.01  # aperture size for depth of field (0 = pinhole camera, larger  = more blur)
focus_dist = np.linalg.norm(lookfrom - lookat) # distance from camera to the plane in focus (used for depth of field calculations)

tile_size = 32    # pixels per tile side — smaller = better load balance, more overhead
num_workers = 16  # number of worker processes; increase for more parallelism, decrease to save memory
starmap_chunksize = 4  # tiles per IPC message — higher = less overhead, lower = better load balance

@njit(cache=True)
def lerp(a, b, t):
    # Blend between a and b
    return a + t * (b - a)

@njit(cache=True)
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

@njit(cache=True)
def _aabb_hit(bbox_min: np.ndarray, bbox_max: np.ndarray, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float) -> bool:
    """Slab test: returns True if ray intersects the axis-aligned bounding box."""
    for i in range(3):
        inv_d = (1.0 / ray_d[i]) if abs(ray_d[i]) > 1e-10 else 1e30
        t0 = (bbox_min[i] - ray_o[i]) * inv_d
        t1 = (bbox_max[i] - ray_o[i]) * inv_d
        if inv_d < 0:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max <= t_min:
            return False
    return True

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


class Plane:
    def __init__(self, point: np.ndarray, normal: np.ndarray, colour_multiplier=0.9):
        self.center = point
        self.normal = normalize(normal)
        self.colour_multiplier = colour_multiplier
        self.material = None

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float) -> float:
        denom = np.dot(ray_d, self.normal)
        if abs(denom) < 1e-6:
            return -1
        t = np.dot(self.center - ray_o, self.normal) / denom
        return t if t_min <= t <= t_max else -1

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        return self.normal

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        if (int(np.floor(hit_point[0])) + int(np.floor(hit_point[2]))) % 2 == 0:
            return np.array([0.8, 0.8, 0.8])
        return np.array([0.1, 0.1, 0.1])

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        random_vec = normalize(np.random.randn(3).astype(np.float64))
        if np.dot(random_vec, normal) < 0:
            random_vec = -random_vec
        scatter_direction = normal + random_vec
        if np.linalg.norm(scatter_direction) < 1e-8:
            scatter_direction = normal
        return scatter_direction


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, colour: np.ndarray,
                 material=None, metal_fuzz=0.0, emission_intensity=8.0,
                 refraction_index=1.5, colour_multiplier=0.9):
        self.center = center
        self.radius = radius
        self.colour = colour
        self.material = material
        self.metal_fuzz = metal_fuzz
        self.emission_intensity = emission_intensity
        self.refraction_index = refraction_index
        self.colour_multiplier = colour_multiplier
        self._last_normal = np.array([0.0, 1.0, 0.0])

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float) -> float:
        oc = ray_o - self.center
        a = np.dot(ray_d, ray_d)
        b = 2.0 * np.dot(oc, ray_d)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return -1
        sqrt_d = np.sqrt(discriminant)
        t = (-b - sqrt_d) / (2 * a)
        if t < t_min or t > t_max:
            t = (-b + sqrt_d) / (2 * a)
            if t < t_min or t > t_max:
                return -1
        hit_point = ray_o + t * ray_d
        outward_normal = normalize(hit_point - self.center)
        self._last_normal = outward_normal if np.dot(ray_d, outward_normal) < 0 else -outward_normal
        return t

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        return self._last_normal

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        if self.material == "emissive":
            return self.colour * self.emission_intensity
        if self.material == "absorbing":
            return np.zeros(3, dtype=np.float64)
        return self.colour

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        if self.material in ("absorbing", "emissive"):
            return None
        if self.material == "metal":
            d = normalize(ray_d)
            reflected = d - 2 * np.dot(d, normal) * normal
            reflected += self.metal_fuzz * normalize(np.random.randn(3))
            return reflected if np.dot(reflected, normal) > 0 else None
        if self.material == "glass":
            d = ray_d
            n = normal if np.dot(d, normal) < 0 else -normal
            eta = 1.0 / self.refraction_index if np.dot(d, normal) < 0 else self.refraction_index
            cos_theta = min(np.dot(-d, n), 1.0)
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))
            r0 = ((1 - eta) / (1 + eta)) ** 2
            reflectance = r0 + (1 - r0) * (1 - cos_theta) ** 5
            if eta * sin_theta > 1.0 or np.random.random() < reflectance:
                return d - 2 * np.dot(d, n) * n
            d_perp = eta * (d + cos_theta * n)
            d_parallel = -np.sqrt(abs(1.0 - np.dot(d_perp, d_perp))) * n
            return d_perp + d_parallel
        random_vec = normalize(np.random.randn(3).astype(np.float64))
        if np.dot(random_vec, normal) < 0:
            random_vec = -random_vec
        scatter_direction = normal + random_vec
        if np.linalg.norm(scatter_direction) < 1e-8:
            scatter_direction = normal
        return scatter_direction


class World:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def add_object(self, obj):
        self.objects.append(obj)

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float = 0.001, t_max: float = float('inf')) -> tuple:
        closest_t = float('inf')
        hit_obj = None

        for obj in self.objects:
            t = obj.hit(ray_o, ray_d, t_min, t_max)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_obj = obj

        return closest_t, hit_obj


# ─────────────────────────── Triangle / Mesh ────────────────────────────────

class Triangle:
    """A single triangle primitive. Uses Möller-Trumbore intersection."""

    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                 colour: np.ndarray, material=None, metal_fuzz=0.0,
                 emission_intensity=5.0, refraction_index=1.5, colour_multiplier=0.9,
                 n0=None, n1=None, n2=None):
        self.v0, self.v1, self.v2 = v0, v1, v2
        self.edge1 = v1 - v0
        self.edge2 = v2 - v0
        self._normal = normalize(np.cross(self.edge1, self.edge2))
        self.n0, self.n1, self.n2 = n0, n1, n2  # vertex normals; None = use flat face normal
        self._last_u = 0.0                        # barycentric coords saved at hit time
        self._last_v = 0.0
        self.colour = colour
        self.material = material
        self.metal_fuzz = metal_fuzz
        self.emission_intensity = emission_intensity
        self.refraction_index = refraction_index
        self.colour_multiplier = colour_multiplier

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float = 0.001, t_max: float = float('inf')) -> float:
        h = np.cross(ray_d, self.edge2)
        a = np.dot(self.edge1, h)
        if abs(a) < 1e-8:          # ray is parallel to triangle
            return -1
        f = 1.0 / a
        s = ray_o - self.v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return -1
        q = np.cross(s, self.edge1)
        v = f * np.dot(ray_d, q)
        if v < 0.0 or u + v > 1.0:
            return -1
        t = f * np.dot(self.edge2, q)
        if t < t_min or t > t_max:
            return -1
        self._last_u = u  # save for get_normal()
        self._last_v = v
        return t

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        if self.n0 is not None:
            # Interpolate vertex normals using barycentric coords from the hit
            # (1-u-v) is the weight of v0's corner, u of v1's, v of v2's
            n = (1 - self._last_u - self._last_v) * self.n0 + self._last_u * self.n1 + self._last_v * self.n2
            n = normalize(n)
        else:
            n = self._normal  # flat face normal fallback
        return n if np.dot(ray_d, n) < 0 else -n

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        if self.material == "emissive":
            return self.colour * self.emission_intensity
        if self.material == "absorbing":
            return np.zeros(3, dtype=np.float64)
        return self.colour

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        if self.material == "metal":
            d = normalize(ray_d)
            reflected = d - 2 * np.dot(d, normal) * normal
            reflected += self.metal_fuzz * normalize(np.random.randn(3))
            return reflected if np.dot(reflected, normal) > 0 else None
        if self.material in ("absorbing", "emissive"):
            return None
        if self.material == "glass":
            d = ray_d
            n = normal if np.dot(d, normal) < 0 else -normal
            eta = 1.0 / self.refraction_index if np.dot(d, normal) < 0 else self.refraction_index
            cos_theta = min(np.dot(-d, n), 1.0)
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))
            # Schlick Fresnel: reflect more at grazing angles
            r0 = ((1 - eta) / (1 + eta)) ** 2
            reflectance = r0 + (1 - r0) * (1 - cos_theta) ** 5
            if eta * sin_theta > 1.0 or np.random.random() < reflectance:
                return d - 2 * np.dot(d, n) * n  # TIR or Fresnel reflect
            d_perp = eta * (d + cos_theta * n)
            d_parallel = -np.sqrt(abs(1.0 - np.dot(d_perp, d_perp))) * n
            return d_perp + d_parallel
        # Diffuse
        random_vec = normalize(np.random.randn(3).astype(np.float64))
        if np.dot(random_vec, normal) < 0:
            random_vec = -random_vec
        scatter_direction = normal + random_vec
        if np.linalg.norm(scatter_direction) < 1e-8:
            scatter_direction = normal
        return scatter_direction


def _box_sa(bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Surface area of an AABB — used by SAH cost function."""
    d = bmax - bmin
    return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])


class BVHNode:
    """Recursive binary BVH. SAH split along the best axis."""

    def __init__(self, triangles: list, depth: int = 0, max_depth: int = 40, leaf_size: int = 4):
        # Per-triangle AABB — reused for both bbox computation and SAH scans
        v0 = np.array([t.v0 for t in triangles])
        v1 = np.array([t.v1 for t in triangles])
        v2 = np.array([t.v2 for t in triangles])
        tri_min = np.minimum(np.minimum(v0, v1), v2)
        tri_max = np.maximum(np.maximum(v0, v1), v2)

        self.bbox_min = tri_min.min(axis=0) - 1e-4
        self.bbox_max = tri_max.max(axis=0) + 1e-4
        self.children = None  # only set by build_parallel

        n = len(triangles)
        if n <= leaf_size or depth >= max_depth:
            self.is_leaf = True
            self.triangles = triangles
            return

        self.is_leaf = False
        centroids = (v0 + v1 + v2) / 3.0
        parent_sa = _box_sa(self.bbox_min, self.bbox_max)

        best_cost  = float('inf')
        best_axis  = -1
        best_idx   = -1  # left partition = [0..best_idx], right = [best_idx+1..]
        best_order = None

        for axis in range(3):
            order = np.argsort(centroids[:, axis])
            s_min = tri_min[order]
            s_max = tri_max[order]

            # Prefix scan: growing left bbox
            l_min = np.minimum.accumulate(s_min, axis=0)
            l_max = np.maximum.accumulate(s_max, axis=0)

            # Suffix scan: growing right bbox
            r_min = np.minimum.accumulate(s_min[::-1], axis=0)[::-1]
            r_max = np.maximum.accumulate(s_max[::-1], axis=0)[::-1]

            # Vectorized SAH cost for all n-1 split positions
            d_l = l_max[:n-1] - l_min[:n-1]
            sa_l = 2.0 * (d_l[:, 0]*d_l[:, 1] + d_l[:, 1]*d_l[:, 2] + d_l[:, 2]*d_l[:, 0])
            d_r = r_max[1:] - r_min[1:]
            sa_r = 2.0 * (d_r[:, 0]*d_r[:, 1] + d_r[:, 1]*d_r[:, 2] + d_r[:, 2]*d_r[:, 0])

            counts_l = np.arange(1, n, dtype=np.float64)
            counts_r = np.arange(n - 1, 0, -1, dtype=np.float64)
            costs = (sa_l * counts_l + sa_r * counts_r) / parent_sa

            idx = int(np.argmin(costs))
            if costs[idx] < best_cost:
                best_cost  = costs[idx]
                best_axis  = axis
                best_idx   = idx
                best_order = order

        # If splitting costs more than a leaf, make a leaf
        if best_axis < 0 or best_cost >= n:
            self.is_leaf = True
            self.triangles = triangles
            return

        ordered    = [triangles[i] for i in best_order]
        left_tris  = ordered[:best_idx + 1]
        right_tris = ordered[best_idx + 1:]
        self.left  = BVHNode(left_tris,  depth + 1, max_depth, leaf_size)
        self.right = BVHNode(right_tris, depth + 1, max_depth, leaf_size)

    @classmethod
    def build_parallel(cls, vertices, faces, normals, normal_faces, num_workers,
                       colour, material, metal_fuzz, emission_intensity, refraction_index, colour_multiplier):
        """
        Split face indices spatially, send each chunk (just a numpy array of ints)
        to a worker. Workers build Triangle objects locally — nothing expensive
        is ever pickled across processes.
        """
        print(f"[BVH] Splitting {len(faces):,} triangles into {num_workers} chunks...")

        centroids = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
        extents   = centroids.max(axis=0) - centroids.min(axis=0)
        axis      = int(np.argmax(extents))
        order     = np.argsort(centroids[:, axis])
        sorted_faces        = faces[order]
        sorted_normal_faces = normal_faces[order] if normal_faces is not None else None

        size = len(sorted_faces) // num_workers
        chunks = [
            sorted_faces[i * size : (i + 1) * size if i < num_workers - 1 else len(sorted_faces)]
            for i in range(num_workers)
        ]
        normal_chunks = [
            sorted_normal_faces[i * size : (i + 1) * size if i < num_workers - 1 else len(sorted_normal_faces)]
            for i in range(num_workers)
        ] if sorted_normal_faces is not None else [None] * num_workers

        print(f"[BVH] Building {num_workers} sub-trees in parallel...")

        args = [(vertices, chunk, normals, nchunk, colour, material, metal_fuzz,
                 emission_intensity, refraction_index, colour_multiplier)
                for chunk, nchunk in zip(chunks, normal_chunks)]

        with mp.Pool(processes=num_workers) as pool:
            subtrees = pool.map(_build_bvh_subtree, args)

        root = cls.__new__(cls)
        root.bbox_min = np.min([s.bbox_min for s in subtrees], axis=0)
        root.bbox_max = np.max([s.bbox_max for s in subtrees], axis=0)
        root.is_leaf  = False
        root.children = subtrees
        root.left     = None
        root.right    = None
        print(f"[BVH] Done.")
        return root

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float):
        """Iterative stack-based BVH traversal. Returns (t, triangle) or (-1, None)."""
        best_t, best_tri = -1, None
        stack = [self]

        while stack:
            node = stack.pop()
            # Tighten the far bound once we have a hit — prunes more of the tree
            far = best_t if best_t > 0 else t_max
            if not _aabb_hit(node.bbox_min, node.bbox_max, ray_o, ray_d, t_min, far):
                continue
            if node.is_leaf:
                for tri in node.triangles:
                    t = tri.hit(ray_o, ray_d, t_min, far)
                    if t > 0 and (best_t < 0 or t < best_t):
                        best_t, best_tri = t, tri
            elif node.children is not None:
                # Parallel-built root: fan out to N sub-trees
                stack.extend(node.children)
            else:
                stack.append(node.left)
                stack.append(node.right)

        return best_t, best_tri


def _build_bvh_subtree(args):
    """
    Receives a tuple of (vertices, face_chunk, normals, normal_face_chunk, ...).
    Builds Triangle objects locally inside the worker, then builds the BVH.
    This avoids ever pickling Triangle objects across processes.
    """
    vertices, face_chunk, normals, normal_face_chunk, colour, material, metal_fuzz, emission_intensity, refraction_index, colour_multiplier = args
    warmup()
    triangles = []
    for i, f in enumerate(face_chunk):
        n0 = n1 = n2 = None
        if normals is not None and normal_face_chunk is not None:
            nf = normal_face_chunk[i]
            if nf[0] >= 0 and nf[1] >= 0 and nf[2] >= 0:
                n0, n1, n2 = normals[nf[0]], normals[nf[1]], normals[nf[2]]
        triangles.append(Triangle(
            vertices[f[0]], vertices[f[1]], vertices[f[2]],
            colour=colour, material=material, metal_fuzz=metal_fuzz,
            emission_intensity=emission_intensity,
            refraction_index=refraction_index,
            colour_multiplier=colour_multiplier,
            n0=n0, n1=n1, n2=n2,
        ))
    return BVHNode(triangles)


class Mesh:
    """
    Loads a Wavefront .obj file and renders it as a collection of triangles
    accelerated by a BVH.

    Usage:
        scene.add_object(Mesh(
            "model.obj",
            colour=np.array([0.8, 0.6, 0.4]),
            material=None,          # diffuse; or "metal", "glass", "emissive", "absorbing"
            scale=0.5,
            translate=np.array([0.0, 0.0, -2.0]),
        ))
    """

    def __init__(self, filepath: str, colour: np.ndarray, material=None,
                 metal_fuzz=0.0, emission_intensity=5.0, refraction_index=1.5,
                 colour_multiplier=0.9, scale=1.0, translate=None):
        self.colour = colour
        self.material = material
        self.colour_multiplier = colour_multiplier
        self._last_tri = None

        vertices, faces, normals, normal_faces = self._load_obj(
            filepath, scale, np.zeros(3) if translate is None else translate,
        )
        smooth = normal_faces is not None
        print(f"[Mesh] {filepath}: {len(faces):,} triangles ({'smooth' if smooth else 'flat'} normals)")
        self.bvh = BVHNode.build_parallel(
            vertices, faces, normals, normal_faces, num_workers,
            colour=colour, material=material, metal_fuzz=metal_fuzz,
            emission_intensity=emission_intensity, refraction_index=refraction_index,
            colour_multiplier=colour_multiplier,
        )

    @staticmethod
    def _load_obj(filepath, scale, translate):
        """Returns (vertices, faces, normals, normal_faces).
        normals / normal_faces are None if the .obj has no vertex normals.
        Triangle creation is deferred to the BVH worker processes."""
        print(f"[Mesh] Reading {filepath}...")
        with open(filepath) as fh:
            lines = fh.readlines()

        # ── Vertices (vectorized) ────────────────────────────────────────────
        print(f"[Mesh] Parsing vertices...")
        v_data = ' '.join(l[2:] for l in lines if len(l) > 2 and l[0] == 'v' and l[1] == ' ')
        vertices = np.fromstring(v_data, sep=' ', dtype=np.float64).reshape(-1, 3)
        vertices = vertices * scale + translate

        # ── Vertex normals (vectorized) ──────────────────────────────────────
        # vn lines look like: "vn 0.123 0.456 0.789"
        vn_lines = [l for l in lines if len(l) > 3 and l[:2] == 'vn']
        if vn_lines:
            vn_data = ' '.join(l[3:] for l in vn_lines)
            normals = np.fromstring(vn_data, sep=' ', dtype=np.float64).reshape(-1, 3)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(norms > 0, norms, 1.0)  # normalise
        else:
            normals = None

        # ── Faces → position + normal index triples ──────────────────────────
        # Each vertex in a face entry can be "pos", "pos/uv", "pos/uv/normal",
        # or "pos//normal". We extract position index (always) and normal index
        # (third component if present).
        print(f"[Mesh] Parsing faces...")
        face_list, normal_face_list = [], []
        has_normals = False

        for line in lines:
            if len(line) < 2 or line[0] != 'f' or line[1] != ' ':
                continue
            parts = line.split()[1:]
            pos_idx, norm_idx = [], []
            for p in parts:
                comp = p.split('/')
                pos_idx.append(int(comp[0]) - 1)
                if len(comp) >= 3 and comp[2]:
                    norm_idx.append(int(comp[2]) - 1)
                    has_normals = True
                else:
                    norm_idx.append(-1)
            for i in range(1, len(pos_idx) - 1):
                face_list.append((pos_idx[0], pos_idx[i], pos_idx[i + 1]))
                normal_face_list.append((norm_idx[0], norm_idx[i], norm_idx[i + 1]))

        faces = np.array(face_list, dtype=np.int32)
        normal_faces = np.array(normal_face_list, dtype=np.int32) if (has_normals and normals is not None) else None
        return vertices, faces, normals, normal_faces

    # ── World interface ──────────────────────────────────────────────────────

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float = 0.001, t_max: float = float('inf')) -> float:
        t, tri = self.bvh.hit(ray_o, ray_d, t_min, t_max)
        self._last_tri = tri
        return t if t > 0 else -1

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        return self._last_tri.get_normal(ray_d, t) if self._last_tri else np.array([0.0, 1.0, 0.0])

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        return self._last_tri.object_colour(hit_point) if self._last_tri else self.colour

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        return self._last_tri.reflect(ray_d, normal) if self._last_tri else None


# ────────────────────────────────────────────────────────────────────────────

def ray_colour(t_min, t_max, ray_o, ray_d, world, depth=0):
    if depth >= depth_limit:
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
    framebuffer = np.frombuffer(shared_array.get_obj()).reshape((img_height, img_width, 3))
    world = shared_world

    j_end = min(j_start + tile_size, img_height)
    i_end = min(i_start + tile_size, img_width)

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

                offset = aperture * radius * (np.cos(angle) * u + np.sin(angle) * v)  # random point in aperture disk

                ray_o = camera_center + offset

                # the ray direction is from the camera toward that pixel's 3D position
                # this is what makes each pixel look in a slightly different direction
                pixel_direction = normalize(pixel_center - camera_center)
                focal_point = camera_center + focus_dist * pixel_direction  # fixed point in space
                ray_d = normalize(focal_point - ray_o)  # ray from lens point to focal point

                colour = ray_colour(0.008, float('inf'), ray_o, ray_d, world, depth=0)

                framebuffer[j, i] += colour  # accumulate the colour for anti-aliasing

            framebuffer[j, i] /= antialising_samples  # average the samples for anti-aliasing

def render_core(world=None, shared=None):
    # forward axis — direction the camera looks
    w = normalize(lookfrom - lookat)

    # right axis — perpendicular to forward and up
    u = normalize(np.cross(vup, w))

    # up axis — perpendicular to both (true up relative to camera)
    v = np.cross(w, u)

    # how much to step in 3D space to move one pixel
    pixel_delta_u = u * (viewport_width / img_width)     # right
    pixel_delta_v = -v * (viewport_height / img_height)  # down

    # find the 3D position of the top-left corner of the viewport
    # start at camera, go forward (focal_length in -Z), then go left and up by half the viewport
    viewport_upper_left = (
        camera_center
        - focal_length * w
        - (viewport_width / 2) * u
        + (viewport_height / 2) * v
    )

    # pixel00_loc is the center of the top-left pixel (not the corner of the viewport)
    # we offset by half a pixel in both directions to center within the pixel
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

    tiles = [
        (j, i, antialising_samples, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, u, v)
        for j in range(0, img_height, tile_size)
        for i in range(0, img_width, tile_size)
    ]
    print(f"[Render] Spawning {min(num_workers, len(tiles))} workers for {len(tiles)} tiles ({img_width}x{img_height}, {antialising_samples} spp)...")
    with mp.Pool(processes=min(num_workers, len(tiles)), initializer=init_worker, initargs=(world, shared)) as pool:
        pool.starmap(render_worker, tiles, chunksize=starmap_chunksize)
    print(f"[Render] All tiles done.")


shared_world = None
shared_array = None
def render(world):
    print(f"[Render] Initialising framebuffer ({img_width}x{img_height})...")
    shared = mp.Array('d', img_height * img_width * 3)
    framebuffer = np.frombuffer(shared.get_obj()).reshape((img_height, img_width, 3))

    # Preview loop runs in a thread so the Pool stays on the main thread (required on Windows)
    stop_preview = threading.Event()
    def preview_loop():
        while not stop_preview.is_set():
            # Quick gamma preview so you can see progress during render
            frame = (np.sqrt(np.clip(framebuffer, 0, 1)) * 255).astype(np.uint8)
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
    np.save(hdr_save_path, framebuffer.copy())
    print(f"[Render] Applying post-processing...")
    out = apply_effect(framebuffer)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[Render] Saved {save_path}")

def warmup():
    # JIT warmup — compile all @njit functions before rendering begins
    _v = np.array([1.0, 2.0, 3.0])
    normalize(_v)
    lerp(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.7, 1.0]), 0.5)
    _aabb_hit(np.zeros(3), np.ones(3), np.zeros(3), np.array([1.0, 1.0, 1.0]), 0.0, 1e30)
    background(_v)

def init_worker(world, array):
    global shared_world, shared_array
    shared_world = world
    shared_array = array
    warmup()  # JIT compile the functions in the worker process

if __name__ == "__main__":
    t_start = time.perf_counter()
    warmup()  # compile on main process too (for BVH build phase)
    scene = World()
    scene.add_object(Plane(
        point=np.array([0.0, -1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
    ))
    scene.add_object(Mesh(
        "models/geodesic/geodesic_classI_3.obj",
        colour=np.array([0.87, 0.19, 0.39]),
        material=None,
        scale=2.0,
        translate=np.array([0.0, 0.6, -2.0]),
    ))
    scene.add_object(Sphere(
        center=np.array([0.0, 2.8, -2.0]),
        radius=0.35,
        colour=np.array([1.0, 0.9, 0.6]),
        material="emissive",
        emission_intensity=8.0,
    ))
    render(world=scene)
    print(f"[Done] Total time: {time.perf_counter() - t_start:.1f}s ({(time.perf_counter() - t_start) / 60:.1f} min)")