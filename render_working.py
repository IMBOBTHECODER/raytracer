import objects
import mesh
import time
import argparse
import numpy as np


def _prompt_vec3(prompt, current):
    """Ask for 3 floats. Press Enter to keep current value."""
    s = input(f"  {prompt} [{current[0]:.2f} {current[1]:.2f} {current[2]:.2f}]: ").strip()
    if not s:
        return current
    try:
        vals = [float(x) for x in s.split()]
        if len(vals) == 3:
            return np.array(vals, dtype=np.float32)
    except ValueError:
        pass
    print("  (invalid — keeping current)")
    return current


def _prompt_float(prompt, current):
    s = input(f"  {prompt} [{current}]: ").strip()
    if not s:
        return current
    try:
        return float(s)
    except ValueError:
        print("  (invalid — keeping current)")
        return current


def _prompt_int(prompt, current):
    s = input(f"  {prompt} [{current}]: ").strip()
    if not s:
        return current
    try:
        return int(s)
    except ValueError:
        print("  (invalid — keeping current)")
        return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .obj file")
    parser.add_argument("--scale",     type=float, default=0.1)
    parser.add_argument("--colour",    type=float, nargs=3, default=[0.192, 0.549, 0.906], metavar=("R", "G", "B"))
    parser.add_argument("--material",  type=str,   default=None, choices=[None, "metal", "glass", "emissive", "absorbing"])
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, -0.55, -2.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--output",    type=str,   default="img/render",
                        help="Base path for output files (e.g. img/render → img/render_001.png)")
    args = parser.parse_args()

    # Import renderer here — not at the top — so multiprocessing workers
    # spawned by mesh.py's BVH builder don't trigger ti.init()
    import renderer
    from config import Config

    t_start = time.perf_counter()
    scene = objects.World()
    scene.add_object(objects.Plane(
        point=np.array([0.0, -1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
    ))
    scene.add_object(mesh.Mesh(
        args.model,
        colour=np.array(args.colour),
        material=args.material,
        scale=args.scale,
        translate=np.array(args.translate),
    ))
    scene.add_object(objects.Sphere(
        center=np.array([8.0, 10.0, 4.0]),
        radius=2.0,
        colour=np.array([1.0, 0.95, 0.8]),
        material="emissive",
        emission_intensity=12.0,
    ))
    renderer.build(scene)

    render_idx = 0
    while True:
        render_idx += 1
        Config.save_path     = f"{args.output}_{render_idx:03d}.png"
        Config.hdr_save_path = f"{args.output}_{render_idx:03d}.npy"

        renderer.run()
        elapsed = time.perf_counter() - t_start
        print(f"[Done] Render #{render_idx} | Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        again = input("\nRender again? [y/N]: ").strip().lower()
        if again != 'y':
            break

        print("New settings (press Enter to keep current):")
        Config.lookfrom = _prompt_vec3("lookfrom (camera pos)    x y z", Config.lookfrom)
        Config.lookat   = _prompt_vec3("lookat   (target pos)    x y z", Config.lookat)
        Config.antialising_samples = _prompt_int("samples per pixel", Config.antialising_samples)
        Config.aperture = _prompt_float("aperture", Config.aperture)

        # Recompute derived values that depend on lookfrom / lookat
        Config.camera_center = Config.lookfrom
        Config.focus_dist = float(np.linalg.norm(Config.lookfrom - Config.lookat))