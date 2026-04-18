import objects
import mesh
import time
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .obj or .fbx file")
    parser.add_argument("--scale",     type=float, default=0.1)
    parser.add_argument("--colour",    type=float, nargs=3, default=[0.192, 0.549, 0.906], metavar=("R", "G", "B"))
    parser.add_argument("--roughness",    type=float, default=1.0)
    parser.add_argument("--metalness",    type=float, default=0.0)
    parser.add_argument("--transmission", type=float, default=0.0)
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, -0.55, -2.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--output",    type=str,   default="img/render",
                        help="Base path for output files")
    parser.add_argument("--lookfrom",  type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Single camera position (overrides config.py)")
    parser.add_argument("--lookat",    type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Camera target")
    parser.add_argument("--samples",   type=int,   default=None,
                        help="Samples per pixel (overrides config.py)")
    parser.add_argument("--views",     type=str,   nargs="+", default=None,
                        metavar="\"X Y Z\"",
                        help="Multiple lookfrom positions — renders one image per view, "
                             "BVH built only once. Use with --lookat for the shared target. "
                             "Example: --views \"5 2 8\" \"10 3 0\" \"-5 2 8\"")
    parser.add_argument("--ignore-fbx-materials", action="store_true",
                        help="Ignore FBX material data and use --colour/--material instead")
    parser.add_argument("--bvh",       type=str,   default=None,
                        metavar="PATH",
                        help="Path to a .npz scene cache. If the file exists, skip BVH "
                             "build and load directly. If it doesn't exist, build and save it.")
    args = parser.parse_args()

    import renderer
    from config import Config

    if args.samples is not None:
        Config.antialising_samples = args.samples

    t_start = time.perf_counter()

    if args.bvh and __import__("os").path.exists(args.bvh):
        # Fast path — load pre-built scene from disk
        renderer.load_scene(args.bvh)
    else:
        # Build from scratch
        scene = objects.World()
        scene.add_object(objects.Plane(
            point=np.array([0.0, -1.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
        ))
        scene.add_object(mesh.Mesh(
            args.model,
            colour=np.array(args.colour),
            scale=args.scale,
            translate=np.array(args.translate),
            ignore_fbx_materials=args.ignore_fbx_materials,
        ))
        scene.add_object(objects.Sphere(
            center=np.array([8.0, 10.0, 4.0]),
            radius=2.0,
            colour=np.array([1.0, 0.95, 0.8]),
            material="emissive",
            emission_intensity=3.0,
        ))
        renderer.build(scene)
        if args.bvh:
            renderer.save_scene(args.bvh)

    print(f"[Setup] Done in {time.perf_counter() - t_start:.1f}s")

    # Build list of (lookfrom, lookat) pairs to render
    lookat = np.array(args.lookat, dtype=np.float32) if args.lookat is not None else Config.lookat

    if args.views:
        camera_list = []
        for v in args.views:
            vals = [float(x) for x in v.split()]
            if len(vals) != 3:
                raise ValueError(f"--views entry must be 3 floats, got: {v!r}")
            camera_list.append(np.array(vals, dtype=np.float32))
    else:
        lookfrom = np.array(args.lookfrom, dtype=np.float32) if args.lookfrom is not None else Config.lookfrom
        camera_list = [lookfrom]

    for idx, lookfrom in enumerate(camera_list, start=1):
        Config.lookfrom      = lookfrom
        Config.lookat        = lookat
        Config.camera_center = lookfrom
        Config.focus_dist    = float(np.linalg.norm(lookfrom - lookat))

        suffix = f"_{idx:03d}" if len(camera_list) > 1 else ""
        Config.save_path     = f"{args.output}{suffix}.png"
        Config.hdr_save_path = f"{args.output}{suffix}.npy"

        t_render = time.perf_counter()
        renderer.run()
        print(f"[Render {idx}/{len(camera_list)}] "
              f"lookfrom={lookfrom.tolist()} | "
              f"{time.perf_counter() - t_render:.1f}s | "
              f"saved {Config.save_path}")

    elapsed = time.perf_counter() - t_start
    print(f"[Done] Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
