# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# Main renderer — takes a model file and CLI flags
python render_working.py models/bugatti.obj --scale 0.1 --samples 128

# Skip BVH rebuild by caching to disk
python render_working.py models/bugatti.obj --bvh scene.npz

# Multiple camera views in one pass (BVH built once)
python render_working.py models/bugatti.obj --views "5 2 8" "10 3 0"
```

Output is saved to `img/render_final.png` and `hdr/render_final.npy` (raw HDR float32).

## Dependencies

```bash
pip install numpy opencv-python taichi pyassimp
# Optional for .glb/.gltf embedded textures:
pip install pygltflib
# Optional for .blend files:
# Blender must be installed and on PATH
```

## File Map

| File | Role |
|------|------|
| `render_working.py` | CLI entrypoint — parses args, builds scene, calls `renderer.run()` |
| `renderer.py` | GPU kernel, MIS path tracer, BVH traversal, post-processing |
| `mesh.py` | CPU-side mesh loading (OBJ/FBX/GLB/BLEND), SAH BVH build, Triangle |
| `objects.py` | `Plane`, `Sphere`, `World` scene container |
| `config.py` | `Config` dataclass — resolution, camera, spp, paths |

## Architecture

### Rendering Pipeline

1. **Scene build** (`renderer.build()`) — collects triangles from SAH `BVHNode` trees, flattens into a single GPU BVH via `_build_flat_bvh()`, uploads all geometry to Taichi fields
2. **`render()` kernel** — `@ti.kernel` over all pixels; each pixel fires `antialising_samples` rays with stratified jitter (grid when spp is a perfect square)
3. **`ray_colour()`** — iterative MIS path tracer (loop up to `depth_limit`):
   - `scene_hit()` dispatches to BVH triangles → planes → spheres
   - **NEE** (Next Event Estimation): for diffuse surfaces, samples every emissive sphere directly
   - **MIS** (Multiple Importance Sampling): `power_heuristic()` balances NEE and BRDF contributions; `prev_specular` flag ensures camera rays and mirror/glass bounces always include emission at full weight
4. **Post-processing** (`apply_effect()`) — multi-scale bloom with cool/blue tint, ACES filmic tone map (`_tone_map_np()`)

### MIS Implementation

Two-strategy balance between **BRDF sampling** and **light sampling (NEE)**:

- **NEE contribution** (per emissive sphere, diffuse surfaces only):
  `w_light = power_heuristic(lpdf, brdf_pdf_for_light_dir)`
  `colour += throughput * w_light * diffuse_brdf * cos_t * emission / lpdf`

- **BRDF contribution** (when BRDF ray hits a sphere emitter):
  `w_brdf = 1.0` if previous bounce was specular/glass or camera ray (no NEE counterpart)
  `w_brdf = power_heuristic(prev_brdf_pdf, sphere_light_pdf(obj_idx, ...))` if previous was diffuse
  Uses only the hit sphere's pdf — NOT the sum across all lights.

Triangle emitters have no NEE, so BRDF always contributes them at full weight.

### Material System (PBR, per triangle)

Materials are float fields set per face during BVH build:

| Field | Type | Effect |
|-------|------|--------|
| `roughness` | f32 0–1 | Metal fuzz amount |
| `metalness` | f32 0–1 | Blend toward specular reflection |
| `transmission` | f32 0–1 | Blend toward glass/dielectric |
| `emission` | vec3 | Emitted radiance (unweighted) |

`scatter()` chooses branch by `r < transmission` → glass, then `r < transmission + metalness` → metal, else diffuse.

### GPU Data Layout (`renderer.py`)

All geometry lives in global Taichi fields allocated in `build()` or `load_scene()`:

- **Triangles**: `tri_v0/v1/v2`, `tri_colour`, `tri_roughness`, `tri_metalness`, `tri_transmission`, `tri_emission`, `tri_normal`, `tri_n0/n1/n2`, `tri_has_smooth`, `tri_uv0/uv1/uv2`, `tri_tex_id`
- **Texture atlas**: `tex_atlas[tex_id, y, x]` — all textures resized to `TEX_SIZE×TEX_SIZE` (512 px), bilinear sampled
- **BVH**: `bvh_bbox_min/max`, `bvh_left/right`, `bvh_tri_start/end`, `bvh_tri_indices` — iterative stack traversal with early-out (`BVH_STACK_SIZE = 64`)
- **Planes/Spheres**: flat arrays, iterated linearly; `n_planes`/`n_spheres` are Python-scope integers captured at kernel compile time

### CPU BVH (`mesh.py`)

`BVHNode` uses full **SAH** (Surface Area Heuristic) — evaluates all split positions on all 3 axes, picks the one minimising `(sa_left * count_left + sa_right * count_right) / sa_parent`. Falls back to leaf if SAH cost ≥ linear scan. `BVHNode.build_parallel()` splits faces by longest centroid axis, builds sub-trees in parallel via `mp.Pool`, then joins under a single root. The entire tree is then re-flattened by `_build_flat_bvh()` in `renderer.py` for GPU upload.

### Camera

Defined in `Config`: `lookfrom`, `lookat`, `vup`, `focal_length`, `viewport_height`, `aperture`, `focus_dist`. Depth of field uses uniform disk sampling scaled by `aperture`; ray direction targets the focal plane at `focus_dist`.

### Post-processing (`apply_effect()`)

Multi-scale bloom: three Gaussian blurs (51, 101, 251 px) on pixels above 0.6 threshold, blended `0.3/0.4/0.6`, scaled by `0.7`, tinted **cool/blue** `[0.8, 0.9, 1.2]`. Added to linear image before ACES tone map.
