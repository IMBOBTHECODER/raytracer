from config import Config
import numpy as np
import multiprocessing as mp


class Triangle:
    """A single triangle primitive. Stores geometry and material for GPU upload."""
    __slots__ = ('v0', 'v1', 'v2', '_normal', 'n0', 'n1', 'n2',
                 'colour', 'material', 'metal_fuzz', 'emission_intensity', 'refraction_index')

    def __init__(self, v0, v1, v2, normal,
                 colour, material, metal_fuzz, emission_intensity, refraction_index,
                 n0=None, n1=None, n2=None):
        self.v0, self.v1, self.v2 = v0, v1, v2
        self._normal = normal
        self.n0, self.n1, self.n2 = n0, n1, n2
        self.colour = colour
        self.material = material
        self.metal_fuzz = metal_fuzz
        self.emission_intensity = emission_intensity
        self.refraction_index = refraction_index


def _box_sa(bmin: np.ndarray, bmax: np.ndarray) -> float:
    d = bmax - bmin
    return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])


class BVHNode:
    """Recursive binary BVH. SAH split along the best axis."""

    def __init__(self, triangles: list, depth: int = 0, max_depth: int = 40, leaf_size: int = 4):
        v0 = np.array([t.v0 for t in triangles])
        v1 = np.array([t.v1 for t in triangles])
        v2 = np.array([t.v2 for t in triangles])
        tri_min = np.minimum(np.minimum(v0, v1), v2)
        tri_max = np.maximum(np.maximum(v0, v1), v2)

        self.bbox_min = tri_min.min(axis=0) - 1e-4
        self.bbox_max = tri_max.max(axis=0) + 1e-4
        self.children = None

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
        best_idx   = -1
        best_order = None

        for axis in range(3):
            order = np.argsort(centroids[:, axis])
            s_min = tri_min[order]
            s_max = tri_max[order]

            l_min = np.minimum.accumulate(s_min, axis=0)
            l_max = np.maximum.accumulate(s_max, axis=0)
            r_min = np.minimum.accumulate(s_min[::-1], axis=0)[::-1]
            r_max = np.maximum.accumulate(s_max[::-1], axis=0)[::-1]

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
    def build_parallel(cls, vertices, faces, face_normals, normals, normal_faces, num_workers,
                       colour, material, metal_fuzz, emission_intensity, refraction_index):
        print(f"[BVH] Splitting {len(faces):,} triangles into {num_workers} chunks...")

        centroids = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
        extents   = centroids.max(axis=0) - centroids.min(axis=0)
        axis      = int(np.argmax(extents))
        order     = np.argsort(centroids[:, axis])
        sorted_faces        = faces[order]
        sorted_face_normals = face_normals[order]
        sorted_normal_faces = normal_faces[order] if normal_faces is not None else None

        size = len(sorted_faces) // num_workers
        chunks = [
            sorted_faces[i * size : (i + 1) * size if i < num_workers - 1 else len(sorted_faces)]
            for i in range(num_workers)
        ]
        fn_chunks = [
            sorted_face_normals[i * size : (i + 1) * size if i < num_workers - 1 else len(sorted_face_normals)]
            for i in range(num_workers)
        ]
        normal_chunks = [
            sorted_normal_faces[i * size : (i + 1) * size if i < num_workers - 1 else len(sorted_normal_faces)]
            for i in range(num_workers)
        ] if sorted_normal_faces is not None else [None] * num_workers

        print(f"[BVH] Building {num_workers} sub-trees in parallel...")

        args = [(vertices, chunk, fn_chunk, normals, nchunk, colour, material, metal_fuzz,
                 emission_intensity, refraction_index)
                for chunk, fn_chunk, nchunk in zip(chunks, fn_chunks, normal_chunks)]

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


def _build_bvh_subtree(args):
    vertices, face_chunk, fn_chunk, normals, normal_face_chunk, colour, material, metal_fuzz, emission_intensity, refraction_index = args
    n = len(face_chunk)
    triangles = [None] * n
    for i in range(n):
        f  = face_chunk[i]
        n0 = n1 = n2 = None
        if normals is not None and normal_face_chunk is not None:
            nf = normal_face_chunk[i]
            if nf[0] >= 0 and nf[1] >= 0 and nf[2] >= 0:
                n0, n1, n2 = normals[nf[0]], normals[nf[1]], normals[nf[2]]
        triangles[i] = Triangle(
            vertices[f[0]], vertices[f[1]], vertices[f[2]], fn_chunk[i],
            colour=colour, material=material, metal_fuzz=metal_fuzz,
            emission_intensity=emission_intensity, refraction_index=refraction_index,
            n0=n0, n1=n1, n2=n2,
        )
    return BVHNode(triangles)


class Mesh:
    def __init__(self, filepath: str, colour: np.ndarray, material=None,
                 metal_fuzz=0.0, emission_intensity=5.0, refraction_index=1.5,
                 scale=1.0, translate=None):
        self.colour = colour
        self.material = material

        vertices, faces, face_normals, normals, normal_faces = self._load_obj(
            filepath, scale, np.zeros(3) if translate is None else translate,
        )
        smooth = normal_faces is not None
        print(f"[Mesh] {filepath}: {len(faces):,} triangles ({'smooth' if smooth else 'flat'} normals)")
        self.bvh = BVHNode.build_parallel(
            vertices, faces, face_normals, normals, normal_faces, Config.num_workers,
            colour=colour, material=material, metal_fuzz=metal_fuzz,
            emission_intensity=emission_intensity, refraction_index=refraction_index,
        )

    @staticmethod
    def _parse_faces(lines):
        """
        Parse face lines. Tries a fast vectorized path for uniform meshes
        (all tri or all quad, same component format). Falls back to a Python loop.
        Returns (faces, normal_faces) as int32 arrays; normal_faces is None if absent.
        """
        face_lines = [l for l in lines if len(l) > 2 and l[0] == 'f' and l[1] == ' ']
        if not face_lines:
            return np.zeros((0, 3), dtype=np.int32), None

        # Detect format from first line
        sample = face_lines[0].split()[1:]
        n_verts = len(sample)

        if n_verts in (3, 4):
            sv = sample[0]
            if '//' in sv:
                fmt, norm_col, comps = 'double_slash', 2, 3
            elif '/' in sv:
                parts = sv.split('/')
                comps = len(parts)
                fmt = 'slash'
                norm_col = 2 if comps >= 3 else -1
            else:
                fmt, norm_col, comps = 'pos_only', -1, 1

            try:
                # Build one big string and parse all indices at once
                text = ' '.join(l[2:] for l in face_lines)
                if fmt == 'double_slash':
                    text = text.replace('//', ' 0 ')
                elif fmt == 'slash':
                    text = text.replace('/', ' ')

                data = np.fromstring(text, sep=' ', dtype=np.int32) - 1
                data = data.reshape(len(face_lines), n_verts, comps)

                pos = data[:, :, 0]
                nrm = data[:, :, norm_col] if norm_col >= 0 else None

                if n_verts == 3:
                    faces = pos.astype(np.int32)
                    normal_faces = nrm.astype(np.int32) if nrm is not None else None
                else:  # quad → 2 triangles
                    faces = np.concatenate([pos[:, [0,1,2]], pos[:, [0,2,3]]], axis=0).astype(np.int32)
                    if nrm is not None:
                        normal_faces = np.concatenate([nrm[:, [0,1,2]], nrm[:, [0,2,3]]], axis=0).astype(np.int32)
                    else:
                        normal_faces = None

                has_normals = normal_faces is not None and np.any(normal_faces >= 0)
                return faces, (normal_faces if has_normals else None)

            except (ValueError, IndexError):
                pass  # fall through to Python loop

        # Fallback: Python loop handles mixed polygon counts and unusual formats
        face_list, normal_face_list = [], []
        has_normals = False
        for line in face_lines:
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
        normal_faces = np.array(normal_face_list, dtype=np.int32) if has_normals else None
        return faces, normal_faces

    @staticmethod
    def _load_obj(filepath, scale, translate):
        print(f"[Mesh] Reading {filepath}...")
        with open(filepath) as fh:
            lines = fh.readlines()

        # ── Vertices (vectorized) ────────────────────────────────────────────
        print(f"[Mesh] Parsing vertices...")
        v_data = ' '.join(l[2:] for l in lines if len(l) > 2 and l[0] == 'v' and l[1] == ' ')
        vertices = np.fromstring(v_data, sep=' ', dtype=np.float64).reshape(-1, 3)
        vertices = vertices * scale + translate

        # ── Vertex normals (vectorized) ──────────────────────────────────────
        vn_lines = [l for l in lines if len(l) > 3 and l[:2] == 'vn']
        if vn_lines:
            vn_data = ' '.join(l[3:] for l in vn_lines)
            normals = np.fromstring(vn_data, sep=' ', dtype=np.float64).reshape(-1, 3)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(norms > 0, norms, 1.0)
        else:
            normals = None

        # ── Faces ────────────────────────────────────────────────────────────
        print(f"[Mesh] Parsing faces...")
        faces, normal_faces = Mesh._parse_faces(lines)

        # ── Face normals (vectorized — avoids per-triangle normalize() calls) ─
        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        crosses = np.cross(e1, e2)
        norms = np.linalg.norm(crosses, axis=1, keepdims=True)
        face_normals = crosses / np.where(norms > 1e-12, norms, 1.0)

        return vertices, faces, face_normals.astype(np.float64), normals, normal_faces