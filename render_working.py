import objects
import mesh
import time
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .obj file")
    parser.add_argument("--scale",     type=float, default=0.1)
    parser.add_argument("--colour",    type=float, nargs=3, default=[0.192, 0.549, 0.906], metavar=("R", "G", "B"))
    parser.add_argument("--material",  type=str,   default=None, choices=[None, "metal", "glass", "emissive", "absorbing"])
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, -0.55, -2.0], metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    # Import renderer here — not at the top — so multiprocessing workers
    # spawned by mesh.py's BVH builder don't trigger ti.init()
    import renderer

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
    renderer.run()
    elapsed = time.perf_counter() - t_start
    print(f"[Done] Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")