# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# Main renderer (512x512, 64 samples, gradient sky, .obj mesh support)
python render_working.py

# Legacy simple renderer (768x768, 16 samples, gradient sky)
python render.py

# Legacy advanced renderer (256x256, 96 samples, HDRI + direct lighting)
python mistake.py
```

Output is saved to `img/render_final.png`. Intermediate previews are written every second during rendering.

## Dependencies

No `requirements.txt` exists. Required packages:

```bash
pip install numpy opencv-python numba
```

`mistake.py` requires `background.hdr` in the working directory for environment lighting.

## Architecture

| File | Resolution | Samples | Background | Mesh Support |
|------|-----------|---------|------------|--------------|
| `render_working.py` | 512×512 | 64 | Gradient sky | Yes (.obj via BVH) |
| `render.py` | 768×768 | 16 | Gradient sky | No |
| `mistake.py` | 256×256 | 96 | HDRI map | No |

### Rendering Pipeline

1. **Camera** — defined by `lookfrom`, `lookat`, `focal_length`, and `aperture` (depth of field via disk sampling)
2. **Ray generation** — per pixel, `antialising_samples` rays are cast with random jitter per tile
3. **`ray_colour()`** — recursive path tracer: intersects scene, samples material BRDF, recurses up to `depth_limit`
4. **`render_worker()`** — renders one tile (`tile_size x tile_size`); spawned in parallel via `mp.Pool`
5. **Shared memory** — `mp.Array` holds the framebuffer across worker processes
6. **Post-processing** — `apply_effect()`: subtle bloom (cool/blue tint), lifted blacks, teal shadow grade, S-curve contrast, vignette, filmic tone mapping. Uses global `save_path`.

### Material System

Materials are set per-object as a string argument:

- `None` — Lambertian diffuse (cosine-weighted hemisphere sampling)
- `"metal"` — specular reflection with a `metal_fuzz` parameter
- `"glass"` — dielectric with Snell's law refraction and Schlick Fresnel approximation
- `"emissive"` — emits light scaled by `emission_intensity`
- `"absorbing"` — returns black (light sink)

### Class Hierarchy (`render_working.py`)

- `Ray` — origin + direction, `.at(t)` for point along ray
- `World` — list of objects, `.hit()` returns nearest intersection
- `Triangle` — single triangle primitive, Möller-Trumbore intersection, supports all materials
- `BVHNode` — recursive binary BVH (median split, longest axis), accelerates triangle meshes
- `Mesh` — loads a `.obj` file, builds a BVH, exposes the same interface as other objects

No `Plane` or `Sphere` classes in `render_working.py` — scene is mesh-only.

### Performance

Math-heavy helpers (`lerp`, `normalize`, `background`) are JIT-compiled with `@njit(cache=True)` (Numba). Cache is invalidated automatically when those functions change. Rendering is parallelized over tiles using `multiprocessing.Pool` with a shared `mp.Array` framebuffer.

### Scene Configuration (`render_working.py`)

Scene is defined at the bottom under `if __name__ == "__main__"`. Current scene: Bugatti model from `models/bugatti.obj`.

```python
scene = World()
scene.add_object(Mesh(
    "models/bugatti.obj",
    colour=np.array([0.7, 0.7, 0.8]),
    material="metal",
    metal_fuzz=0.05,
    scale=0.01,
    translate=np.array([0.0, -1.0, -2.0]),
))
```

Camera: `lookfrom=[3.0, 0.5, 2.0]`, `lookat=[0.0, -1.0, -2.0]` (3/4 front-corner view).