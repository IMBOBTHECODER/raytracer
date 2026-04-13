from config import Config
import numpy as np
from functions import normalize
from numba import njit
import multiprocessing as mp


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
            vertices, faces, normals, normal_faces, Config.num_workers,
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