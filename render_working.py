import objects
import mesh
import time
import numpy as np

if __name__ == "__main__":
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
        "models/bugatti.obj",
        colour=np.array([0.192, 0.549, 0.906]),
        material=None,
        scale=0.1,
        translate=np.array([0.0, -0.55, -2.0]),  # y: -1 - (-4.5*0.1) = -0.55 sits on floor
    ))
    scene.add_object(objects.Sphere(
        center=np.array([8.0, 10.0, 4.0]),  # off to the side and high up
        radius=2.0,
        colour=np.array([1.0, 0.95, 0.8]),
        material="emissive",
        emission_intensity=12.0,
    ))
    renderer.build(scene)
    renderer.run()
    elapsed = time.perf_counter() - t_start
    print(f"[Done] Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")