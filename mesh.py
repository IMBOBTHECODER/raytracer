from config import Config
import numpy as np
import multiprocessing as mp


def _get_prop(props, name, default):
    """Read a named property from a pyassimp material properties dict.
    Matches: exact key, suffix match ('$clr.diffuse' for 'diffuse'),
    or last-segment match ('metallicFactor' for '$mat.metallicfactor').
    """
    last = name.split('.')[-1]  # e.g. 'metallicfactor' from '$mat.metallicfactor'
    for key_tuple, value in props.items():
        k = key_tuple[0].lower()
        if k == name or k == last or k.endswith('.' + name):
            return value
    return default


def _map_material(mat):
    """
    Extract PBR properties from a pyassimp material.
    Returns (base_colour, roughness, metalness, transmission, emission).
    """
    props = mat.properties
    base_colour  = np.array(_get_prop(props, "diffuse",                [1, 1, 1]), dtype=np.float32).flatten()[:3]
    emissive     = np.array(_get_prop(props, "emissive",               [0, 0, 0]), dtype=np.float32).flatten()[:3]
    opacity      = float(np.array(_get_prop(props, "opacity",          1.0)).flat[0])
    shininess    = float(np.array(_get_prop(props, "shininess",        0.0)).flat[0])
    metalness    = float(np.array(_get_prop(props, "$mat.metallicfactor",  0.0)).flat[0])
    roughness    = float(np.array(_get_prop(props, "$mat.roughnessfactor", 1.0)).flat[0])

    # Fallback: derive from legacy shininess if no PBR props present
    if metalness == 0.0 and roughness == 1.0 and shininess > 0:
        metalness = min(1.0, shininess / 200.0)
        roughness = max(0.05, 1.0 - shininess / 128.0)

    transmission = max(0.0, 1.0 - opacity)
    emission = emissive.astype(np.float32)

    return base_colour, roughness, metalness, transmission, emission


class Triangle:
    """A single triangle primitive. Stores geometry and PBR material for GPU upload."""
    __slots__ = ('v0', 'v1', 'v2', '_normal', 'n0', 'n1', 'n2',
                 'colour', 'roughness', 'metalness', 'transmission', 'emission',
                 'uv0', 'uv1', 'uv2', 'tex_id', 'emis_tex_id')

    def __init__(self, v0, v1, v2, normal,
                 colour, roughness=1.0, metalness=0.0, transmission=0.0,
                 emission=None,
                 n0=None, n1=None, n2=None,
                 uv0=None, uv1=None, uv2=None, tex_id=-1, emis_tex_id=-1):
        self.v0, self.v1, self.v2 = v0, v1, v2
        self._normal = normal
        self.n0, self.n1, self.n2 = n0, n1, n2
        self.colour = colour
        self.roughness = float(roughness)
        self.metalness = float(metalness)
        self.transmission = float(transmission)
        _z3 = np.zeros(3, dtype=np.float32)
        self.emission = emission.astype(np.float32) if emission is not None else _z3
        _z2 = np.zeros(2, dtype=np.float32)
        self.uv0 = uv0 if uv0 is not None else _z2
        self.uv1 = uv1 if uv1 is not None else _z2
        self.uv2 = uv2 if uv2 is not None else _z2
        self.tex_id = tex_id
        self.emis_tex_id = int(emis_tex_id)


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
                       colour, roughness=1.0, metalness=0.0, transmission=0.0, emission=None,
                       face_mats=None, uvs=None, uv_faces=None, face_tex_ids=None,
                       face_emis_tex_ids=None):
        print(f"[BVH] Splitting {len(faces):,} triangles into {num_workers} chunks...")

        centroids = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
        extents   = centroids.max(axis=0) - centroids.min(axis=0)
        axis      = int(np.argmax(extents))
        order     = np.argsort(centroids[:, axis])
        sorted_faces             = faces[order]
        sorted_face_normals      = face_normals[order]
        sorted_normal_faces      = normal_faces[order]      if normal_faces      is not None else None
        sorted_face_mats         = {k: v[order] for k, v in face_mats.items()} if face_mats is not None else None
        sorted_uv_faces          = uv_faces[order]          if uv_faces          is not None else None
        sorted_face_tex_ids      = face_tex_ids[order]      if face_tex_ids      is not None else None
        sorted_face_emis_tex_ids = face_emis_tex_ids[order] if face_emis_tex_ids is not None else None

        size = len(sorted_faces) // num_workers
        def chunk(arr, i):
            lo = i * size
            hi = (i + 1) * size if i < num_workers - 1 else len(sorted_faces)
            return arr[lo:hi]

        print(f"[BVH] Building {num_workers} sub-trees in parallel...")

        args = [
            (vertices,
             chunk(sorted_faces, i),
             chunk(sorted_face_normals, i),
             normals,
             chunk(sorted_normal_faces, i)       if sorted_normal_faces      is not None else None,
             colour, roughness, metalness, transmission, emission,
             {k: chunk(v, i) for k, v in sorted_face_mats.items()} if sorted_face_mats is not None else None,
             uvs,
             chunk(sorted_uv_faces, i)           if sorted_uv_faces          is not None else None,
             chunk(sorted_face_tex_ids, i)       if sorted_face_tex_ids      is not None else None,
             chunk(sorted_face_emis_tex_ids, i)  if sorted_face_emis_tex_ids is not None else None)
            for i in range(num_workers)
        ]

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
    vertices, face_chunk, fn_chunk, normals, normal_face_chunk, colour, roughness, metalness, \
    transmission, emission, face_mats, uvs, uv_face_chunk, face_tex_id_chunk, \
    face_emis_tex_id_chunk = args
    n = len(face_chunk)
    triangles = [None] * n
    _z2 = np.zeros(2, dtype=np.float32)
    _z3 = np.zeros(3, dtype=np.float32)
    for i in range(n):
        f  = face_chunk[i]
        n0 = n1 = n2 = None
        if normals is not None and normal_face_chunk is not None:
            nf = normal_face_chunk[i]
            if nf[0] >= 0 and nf[1] >= 0 and nf[2] >= 0:
                n0, n1, n2 = normals[nf[0]], normals[nf[1]], normals[nf[2]]
        if face_mats is not None:
            c   = face_mats['colour'][i]
            ro  = float(face_mats['roughness'][i])
            me  = float(face_mats['metalness'][i])
            tr  = float(face_mats['transmission'][i])
            em  = face_mats['emission'][i]
        else:
            c, ro, me, tr, em = colour, roughness, metalness, transmission, emission
        if uvs is not None and uv_face_chunk is not None:
            uf  = uv_face_chunk[i]
            uv0 = uvs[uf[0]].astype(np.float32) if uf[0] >= 0 else _z2
            uv1 = uvs[uf[1]].astype(np.float32) if uf[1] >= 0 else _z2
            uv2 = uvs[uf[2]].astype(np.float32) if uf[2] >= 0 else _z2
        else:
            uv0 = uv1 = uv2 = _z2
        tex_id      = int(face_tex_id_chunk[i])      if face_tex_id_chunk      is not None else -1
        emis_tex_id = int(face_emis_tex_id_chunk[i]) if face_emis_tex_id_chunk is not None else -1
        triangles[i] = Triangle(
            vertices[f[0]], vertices[f[1]], vertices[f[2]], fn_chunk[i],
            colour=c, roughness=ro, metalness=me, transmission=tr, emission=em,
            n0=n0, n1=n1, n2=n2,
            uv0=uv0, uv1=uv1, uv2=uv2, tex_id=tex_id, emis_tex_id=emis_tex_id,
        )
    return BVHNode(triangles)


class Mesh:
    def __init__(self, filepath: str, colour: np.ndarray,
                 roughness=1.0, metalness=0.0, transmission=0.0, emission=None,
                 scale=1.0, translate=None, ignore_fbx_materials=False):
        import os, cv2
        self.colour = colour
        _z3 = np.zeros(3, dtype=np.float32)
        self.emission = emission if emission is not None else _z3

        vertices, faces, face_normals, normals, normal_faces, face_mats, uvs, uv_faces, tex_paths, tex_preloaded = \
            self._load_file(filepath, scale, np.zeros(3) if translate is None else translate)

        # Load texture images — only keep ones that actually load; failed → tex_id -1
        basedir = os.path.dirname(os.path.abspath(filepath))
        self.textures = []
        tex_id_remap = {}  # original index in tex_paths → new index in self.textures
        for i, p in enumerate(tex_paths or []):
            img = None
            if p in tex_preloaded:
                img = tex_preloaded[p]
            else:
                full = p if os.path.isabs(p) else os.path.join(basedir, p)
                raw  = cv2.imread(full) if os.path.exists(full) else None
                if raw is not None:
                    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                else:
                    print(f"[Mesh] Warning: texture not found: {full!r} — using material colour")
            if img is not None:
                tex_id_remap[i] = len(self.textures)
                self.textures.append(img)

        # Per-face texture IDs — remap to pruned list; missing ones become -1
        n_faces = len(faces)
        if face_mats is not None and 'tex_id' in face_mats:
            raw_ids      = face_mats['tex_id'].astype(np.int32)
            face_tex_ids = np.full(n_faces, -1, dtype=np.int32)
            for old_id, new_id in tex_id_remap.items():
                face_tex_ids[raw_ids == old_id] = new_id
        elif tex_paths and self.textures:
            face_tex_ids = np.zeros(n_faces, dtype=np.int32)
        else:
            face_tex_ids = np.full(n_faces, -1, dtype=np.int32)

        # Per-face emission texture IDs — same remapping
        if face_mats is not None and 'emis_tex_id' in face_mats:
            raw_emis_ids      = face_mats['emis_tex_id'].astype(np.int32)
            face_emis_tex_ids = np.full(n_faces, -1, dtype=np.int32)
            for old_id, new_id in tex_id_remap.items():
                face_emis_tex_ids[raw_emis_ids == old_id] = new_id
        else:
            face_emis_tex_ids = np.full(n_faces, -1, dtype=np.int32)

        smooth = normal_faces is not None
        print(f"[Mesh] {filepath}: {len(faces):,} triangles ({'smooth' if smooth else 'flat'} normals), "
              f"{len([t for t in self.textures if t is not None])} texture(s)")

        self.bvh = BVHNode.build_parallel(
            vertices, faces, face_normals, normals, normal_faces, Config.num_workers,
            colour=colour, roughness=roughness, metalness=metalness,
            transmission=transmission, emission=self.emission,
            face_mats=None if ignore_fbx_materials else face_mats,
            uvs=uvs, uv_faces=uv_faces, face_tex_ids=face_tex_ids,
            face_emis_tex_ids=face_emis_tex_ids,
        )

    @staticmethod
    def _parse_faces(lines):
        """
        Parse face lines. Returns (faces, normal_faces, uv_faces) as int32 arrays;
        normal_faces and uv_faces are None if absent.
        """
        face_lines = [l for l in lines if len(l) > 2 and l[0] == 'f' and l[1] == ' ']
        if not face_lines:
            return np.zeros((0, 3), dtype=np.int32), None, None

        sample = face_lines[0].split()[1:]
        n_verts = len(sample)

        if n_verts in (3, 4):
            sv = sample[0]
            if '//' in sv:
                fmt, norm_col, uv_col, comps = 'double_slash', 2, -1, 3
            elif '/' in sv:
                parts = sv.split('/')
                comps = len(parts)
                fmt = 'slash'
                norm_col = 2 if comps >= 3 else -1
                uv_col   = 1 if comps >= 2 else -1
            else:
                fmt, norm_col, uv_col, comps = 'pos_only', -1, -1, 1

            try:
                text = ' '.join(l[2:] for l in face_lines)
                if fmt == 'double_slash':
                    text = text.replace('//', ' 0 ')
                elif fmt == 'slash':
                    text = text.replace('/', ' ')

                data = np.fromstring(text, sep=' ', dtype=np.int32) - 1
                data = data.reshape(len(face_lines), n_verts, comps)

                pos = data[:, :, 0]
                nrm = data[:, :, norm_col] if norm_col >= 0 else None
                uvf = data[:, :, uv_col]   if uv_col  >= 0 else None

                if n_verts == 3:
                    faces        = pos.astype(np.int32)
                    normal_faces = nrm.astype(np.int32) if nrm is not None else None
                    uv_faces     = uvf.astype(np.int32) if uvf is not None else None
                else:
                    faces        = np.concatenate([pos[:, [0,1,2]], pos[:, [0,2,3]]], axis=0).astype(np.int32)
                    normal_faces = np.concatenate([nrm[:, [0,1,2]], nrm[:, [0,2,3]]], axis=0).astype(np.int32) if nrm is not None else None
                    uv_faces     = np.concatenate([uvf[:, [0,1,2]], uvf[:, [0,2,3]]], axis=0).astype(np.int32) if uvf is not None else None

                has_normals = normal_faces is not None and np.any(normal_faces >= 0)
                has_uvs     = uv_faces     is not None and np.any(uv_faces     >= 0)
                return faces, (normal_faces if has_normals else None), (uv_faces if has_uvs else None)

            except (ValueError, IndexError):
                pass

        face_list, normal_face_list, uv_face_list = [], [], []
        has_normals = has_uvs = False
        for line in face_lines:
            parts = line.split()[1:]
            pos_idx, norm_idx, uv_idx = [], [], []
            for p in parts:
                comp = p.split('/')
                pos_idx.append(int(comp[0]) - 1)
                if len(comp) >= 2 and comp[1]:
                    uv_idx.append(int(comp[1]) - 1);  has_uvs = True
                else:
                    uv_idx.append(-1)
                if len(comp) >= 3 and comp[2]:
                    norm_idx.append(int(comp[2]) - 1);  has_normals = True
                else:
                    norm_idx.append(-1)
            for i in range(1, len(pos_idx) - 1):
                face_list.append((pos_idx[0], pos_idx[i], pos_idx[i+1]))
                normal_face_list.append((norm_idx[0], norm_idx[i], norm_idx[i+1]))
                uv_face_list.append((uv_idx[0], uv_idx[i], uv_idx[i+1]))

        faces        = np.array(face_list,        dtype=np.int32)
        normal_faces = np.array(normal_face_list, dtype=np.int32) if has_normals else None
        uv_faces     = np.array(uv_face_list,     dtype=np.int32) if has_uvs     else None
        return faces, normal_faces, uv_faces

    @staticmethod
    def _parse_mtl(obj_path, mtllib_name):
        """Returns {mat_name: tex_path} from a .mtl file next to the .obj."""
        import os
        mtl_path = os.path.join(os.path.dirname(os.path.abspath(obj_path)), mtllib_name.strip())
        result = {}
        current = None
        try:
            with open(mtl_path) as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if not parts:
                        continue
                    if parts[0] == 'newmtl' and len(parts) > 1:
                        current = parts[1].strip()
                    elif parts[0] in ('map_Kd', 'map_kd') and current and len(parts) > 1:
                        result[current] = parts[1].strip()
        except FileNotFoundError:
            pass
        return result

    @staticmethod
    def _load_file(filepath, scale, translate):
        ext = filepath.rsplit('.', 1)[-1].lower()
        if ext == 'obj':
            vertices, faces, face_normals, normals, normal_faces, uvs, uv_faces, tex_paths = \
                Mesh._load_obj(filepath, scale, translate)
            return vertices, faces, face_normals, normals, normal_faces, None, uvs, uv_faces, tex_paths, {}
        if ext == 'blend':
            return Mesh._load_blend(filepath, scale, translate)
        return Mesh._load_assimp(filepath, scale, translate)

    @staticmethod
    def _load_blend(filepath, scale, translate):
        import subprocess, tempfile, os, shutil
        blender = shutil.which("blender")
        if blender is None:
            raise RuntimeError("Blender not found. Install with: apt-get install -y blender")
        print(f"[Mesh] Reading {filepath} via Blender subprocess...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_obj = os.path.join(tmpdir, "export.obj")
            script = (
                "import bpy\n"
                "bpy.ops.object.select_all(action='SELECT')\n"
                "try:\n"
                f"    bpy.ops.wm.obj_export(filepath={tmp_obj!r}, export_uv=True, export_normals=True, export_materials=True)\n"
                "except Exception:\n"
                f"    bpy.ops.export_scene.obj(filepath={tmp_obj!r}, use_uvs=True, use_normals=True, use_materials=True)\n"
            )
            script_path = os.path.join(tmpdir, "export.py")
            with open(script_path, "w") as f:
                f.write(script)
            result = subprocess.run(
                [blender, "--background", os.path.abspath(filepath), "--python", script_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Blender failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
            vertices, faces, face_normals, normals, normal_faces, uvs, uv_faces, tex_paths = \
                Mesh._load_obj(tmp_obj, scale, translate)
        return vertices, faces, face_normals, normals, normal_faces, None, uvs, uv_faces, tex_paths, {}

    @staticmethod
    def _load_assimp(filepath, scale, translate):
        import pyassimp
        import pyassimp.postprocess as pp
        print(f"[Mesh] Reading {filepath} via pyassimp...")
        processing = (
            pp.aiProcess_Triangulate |
            pp.aiProcess_GenSmoothNormals |
            pp.aiProcess_JoinIdenticalVertices |
            pp.aiProcess_PreTransformVertices
        )

        all_verts, all_faces, all_norms, all_uvs = [], [], [], []
        all_colours, all_roughness, all_metalness, all_transmission, all_emission, all_tex_ids, all_emis_tex_ids = [], [], [], [], [], [], []
        tex_paths = []
        tex_preloaded = {}
        path_to_id = {}
        vert_offset = 0
        has_uvs = False

        with pyassimp.load(filepath, processing=processing) as scene:
            if not scene.meshes:
                raise ValueError(f"No meshes found in {filepath}")

            # Extract embedded textures (path like "*0", "*1", ...)
            if hasattr(scene, 'textures') and scene.textures:
                import cv2 as _cv2, ctypes as _ct
                for i, etex in enumerate(scene.textures):
                    key = f"*{i}"
                    try:
                        n_bytes = etex.width if etex.height == 0 else etex.width * etex.height * 4
                        # pyassimp wraps pcData as a size-0 numpy array; extract the raw C pointer
                        data_obj = etex.data
                        if isinstance(data_obj, np.ndarray):
                            addr = data_obj.ctypes.data          # int: address of backing C buffer
                        else:
                            try:
                                addr = _ct.cast(data_obj, _ct.c_void_p).value   # ctypes POINTER
                            except TypeError:
                                addr = _ct.addressof(data_obj)   # ctypes array
                        if not addr:
                            raise ValueError("null texture data pointer")
                        raw = _ct.string_at(addr, n_bytes)
                        if etex.height == 0:
                            buf = np.frombuffer(raw, dtype=np.uint8)
                            img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
                            if img is not None:
                                tex_preloaded[key] = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        else:
                            data_arr = np.frombuffer(raw, dtype=np.uint8).reshape(etex.height, etex.width, 4)
                            tex_preloaded[key] = data_arr[:, :, [2, 1, 0]].astype(np.float32) / 255.0
                    except Exception as e:
                        print(f"[Mesh] Warning: embedded texture *{i} failed: {e} "
                              f"(data type={type(etex.data).__name__}, "
                              f"shape={getattr(etex.data, 'shape', 'n/a')}, "
                              f"width={etex.width}, height={etex.height})")

            # Build texture path list from all materials upfront.
            # pyassimp key format varies: tuple ("$tex.file", sem, idx) in older versions,
            # plain string "file" in newer versions.
            # sem: 0/1/12 = diffuse/base-color, 4 = emissive (legacy), 14 = emission color (PBR)
            for mat in scene.materials:
                for key, val in mat.properties.items():
                    name = key[0] if isinstance(key, tuple) else key
                    sem  = key[1] if isinstance(key, tuple) and len(key) > 1 else 0
                    is_tex = (name == 'file' or name.endswith('.file'))
                    if is_tex and sem in (0, 1, 4, 12, 14) and isinstance(val, str) and val not in path_to_id:
                        path_to_id[val] = len(tex_paths)
                        tex_paths.append(val)

            # Build world transform per mesh by walking the node tree
            mesh_ptr_to_idx = {id(m): i for i, m in enumerate(scene.meshes)}
            mesh_transforms = [np.eye(4)] * len(scene.meshes)
            node_stack = [(scene.rootnode, np.eye(4))]
            while node_stack:
                node, parent_transform = node_stack.pop()
                world = parent_transform @ np.array(node.transformation, dtype=np.float64).T
                for mesh_ref in node.meshes:
                    idx = mesh_ref if isinstance(mesh_ref, int) else mesh_ptr_to_idx.get(id(mesh_ref))
                    if idx is not None:
                        mesh_transforms[idx] = world
                for child in node.children:
                    node_stack.append((child, world))

            for mi, m in enumerate(scene.meshes):
                T = mesh_transforms[mi]
                verts_h = np.concatenate([m.vertices, np.ones((len(m.vertices), 1))], axis=1)
                verts   = (verts_h @ T[:3, :].T).astype(np.float64)
                R       = T[:3, :3]
                norms   = (np.array(m.normals, dtype=np.float64) @ R.T)
                faces_m = np.array(m.faces,    dtype=np.int32) + vert_offset
                n_verts = len(verts)
                n_faces = len(faces_m)

                mat     = scene.materials[m.materialindex]
                colour, roughness, metalness, transmission, emission = _map_material(mat)

                # Per-vertex UVs (channel 0)
                if m.texturecoords is not None and len(m.texturecoords) > 0 \
                        and m.texturecoords[0] is not None and len(m.texturecoords[0]) == n_verts:
                    uvs_m    = np.array(m.texturecoords[0], dtype=np.float64)[:, :2]
                    has_uvs  = True
                else:
                    uvs_m = np.zeros((n_verts, 2), dtype=np.float64)

                # Texture ID for this submesh's material (diffuse and emissive)
                tex_id_m      = -1
                emis_tex_id_m = -1
                for key, val in mat.properties.items():
                    name = key[0] if isinstance(key, tuple) else key
                    sem  = key[1] if isinstance(key, tuple) and len(key) > 1 else 0
                    is_tex = (name == 'file' or name.endswith('.file'))
                    if not is_tex or not isinstance(val, str):
                        continue
                    if sem in (0, 1, 12) and tex_id_m == -1:
                        tex_id_m = path_to_id.get(val, -1)
                    elif sem in (4, 14) and emis_tex_id_m == -1:
                        emis_tex_id_m = path_to_id.get(val, -1)

                # If an emission texture exists but the constant emissive is zero, promote to
                # (1,1,1) so the kernel multiplication gives: 1 * texture = texture.
                if emis_tex_id_m >= 0 and emission.max() == 0.0:
                    emission = np.ones(3, dtype=np.float32)

                # Vertex colours: use per-face average when available and material has no real colour
                raw_faces = np.array(m.faces, dtype=np.int32)
                if (m.colors is not None and len(m.colors) > 0
                        and m.colors[0] is not None and len(m.colors[0]) == n_verts):
                    vc = np.array(m.colors[0], dtype=np.float32)[:, :3]
                    face_col = (vc[raw_faces[:, 0]] + vc[raw_faces[:, 1]] + vc[raw_faces[:, 2]]) / 3.0
                else:
                    face_col = np.tile(colour, (n_faces, 1)).astype(np.float32)

                all_verts.append(verts)
                all_faces.append(faces_m)
                all_norms.append(norms)
                all_uvs.append(uvs_m)
                all_colours.append(face_col)
                all_roughness.append(     np.full(n_faces, roughness,     dtype=np.float32))
                all_metalness.append(     np.full(n_faces, metalness,     dtype=np.float32))
                all_transmission.append(  np.full(n_faces, transmission,  dtype=np.float32))
                all_emission.append(      np.tile(emission, (n_faces, 1)).astype(np.float32))
                all_tex_ids.append(       np.full(n_faces, tex_id_m,      dtype=np.int32))
                all_emis_tex_ids.append(  np.full(n_faces, emis_tex_id_m, dtype=np.int32))
                vert_offset += n_verts

        vertices = np.concatenate(all_verts, axis=0) * scale + translate
        faces    = np.concatenate(all_faces, axis=0)
        normals  = np.concatenate(all_norms, axis=0)
        uvs      = np.concatenate(all_uvs,   axis=0) if has_uvs else None

        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(lens > 0, lens, 1.0)

        normal_faces = faces.copy()  # assimp gives per-vertex normals; UV faces are same

        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        crosses = np.cross(e1, e2)
        lens = np.linalg.norm(crosses, axis=1, keepdims=True)
        face_normals = crosses / np.where(lens > 1e-12, lens, 1.0)

        face_mats = {
            'colour':       np.concatenate(all_colours,      axis=0).astype(np.float64),
            'roughness':    np.concatenate(all_roughness,    axis=0),
            'metalness':    np.concatenate(all_metalness,    axis=0),
            'transmission': np.concatenate(all_transmission, axis=0),
            'emission':     np.concatenate(all_emission,     axis=0),
            'tex_id':       np.concatenate(all_tex_ids,      axis=0),
            'emis_tex_id':  np.concatenate(all_emis_tex_ids, axis=0),
        }

        # uv_faces == faces: assimp gives per-vertex UVs after JoinIdenticalVertices
        uv_faces = faces.copy() if has_uvs else None

        # For GLB/GLTF: pyassimp's embedded texture API is unreliable; use pygltflib instead
        ext = filepath.rsplit('.', 1)[-1].lower()
        if ext in ('glb', 'gltf') and not tex_preloaded and tex_paths:
            tex_preloaded = Mesh._extract_gltf_textures(filepath)

        return vertices, faces, face_normals.astype(np.float64), normals, normal_faces, face_mats, uvs, uv_faces, tex_paths, tex_preloaded

    @staticmethod
    def _extract_gltf_textures(filepath):
        """Extract embedded images from a GLB/GLTF file using pygltflib.
        Keys are '*N' by TEXTURE index (matching Assimp's '*N' convention),
        not image index — the two differ when textures reference images out of order.
        """
        try:
            from pygltflib import GLTF2
        except ImportError:
            print("[Mesh] pygltflib not installed — run: pip install pygltflib")
            return {}
        import cv2, base64
        result = {}
        try:
            gltf = GLTF2().load(filepath)
            blob = gltf.binary_blob()

            # Load every image by its image index first
            images_by_idx = {}
            for i, image in enumerate(gltf.images or []):
                try:
                    raw = None
                    if image.bufferView is not None and blob is not None:
                        bv  = gltf.bufferViews[image.bufferView]
                        raw = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
                    elif image.uri and image.uri.startswith('data:'):
                        raw = base64.b64decode(image.uri.split(',', 1)[1])
                    if raw:
                        buf = np.frombuffer(raw, dtype=np.uint8)
                        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if img is not None:
                            images_by_idx[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                except Exception as e:
                    print(f"[Mesh] Warning: gltf image {i} failed: {e}")

            # Key by TEXTURE index so '*N' matches what Assimp assigns to material tex paths
            for i, texture in enumerate(gltf.textures or []):
                src = getattr(texture, 'source', None)
                if src is not None and src in images_by_idx:
                    result[f"*{i}"] = images_by_idx[src]

            # Fallback: if no textures array, key directly by image index
            if not result:
                result = {f"*{i}": img for i, img in images_by_idx.items()}

        except Exception as e:
            print(f"[Mesh] Warning: pygltflib load failed: {e}")
        return result

    @staticmethod
    def _load_obj(filepath, scale, translate):
        print(f"[Mesh] Reading {filepath}...")
        with open(filepath) as fh:
            lines = fh.readlines()

        # ── Vertices ─────────────────────────────────────────────────────────
        print(f"[Mesh] Parsing vertices...")
        v_data = ' '.join(l[2:] for l in lines if len(l) > 2 and l[0] == 'v' and l[1] == ' ')
        vertices = np.fromstring(v_data, sep=' ', dtype=np.float64).reshape(-1, 3)
        vertices = vertices * scale + translate

        # ── UV coordinates ───────────────────────────────────────────────────
        vt_lines = [l for l in lines if len(l) > 3 and l[:2] == 'vt']
        if vt_lines:
            vt_data = ' '.join(l[3:] for l in vt_lines)
            uvs = np.fromstring(vt_data, sep=' ', dtype=np.float64).reshape(-1, -1)[:, :2]
        else:
            uvs = None

        # ── Vertex normals ───────────────────────────────────────────────────
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
        faces, normal_faces, uv_faces = Mesh._parse_faces(lines)

        # ── Face normals ─────────────────────────────────────────────────────
        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        crosses = np.cross(e1, e2)
        norms = np.linalg.norm(crosses, axis=1, keepdims=True)
        face_normals = crosses / np.where(norms > 1e-12, norms, 1.0)

        # ── MTL texture paths ────────────────────────────────────────────────
        tex_paths = []
        for l in lines:
            if l.startswith('mtllib') and len(l.split()) > 1:
                mat_tex = Mesh._parse_mtl(filepath, l.split(None, 1)[1])
                tex_paths = list(dict.fromkeys(mat_tex.values()))
                break

        return vertices, faces, face_normals.astype(np.float64), normals, normal_faces, uvs, uv_faces, tex_paths