import renderer
import objects
import mesh
import time

import numpy as np

if __name__ == "__main__":
    t_start = time.perf_counter()
    renderer.warmup()  # compile on main process too (for BVH build phase)
    scene = objects.World()
    scene.add_object(objects.Plane(
        point=np.array([0.0, -1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
    ))
    scene.add_object(mesh.Mesh(
        "models/geodesic/geodesic_classI_1.obj",
        colour=np.array([0.87, 0.19, 0.39]),
        material=None,
        scale=2.0,
        translate=np.array([0.0, 0.6, -2.0]),
    ))
    scene.add_object(objects.Sphere(
        center=np.array([0.0, 2.8, -2.0]),
        radius=0.35,
        colour=np.array([1.0, 0.9, 0.6]),
        material="emissive",
        emission_intensity=8.0,
    ))
    renderer.render(world=scene)
    print(f"[Done] Total time: {time.perf_counter() - t_start:.1f}s ({(time.perf_counter() - t_start) / 60:.1f} min)")