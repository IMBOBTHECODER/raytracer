import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    img_width = 512
    img_height = 512
    aspect_ratio = img_width / img_height

    lookfrom = np.array([3.0, 1.5, 2.0])   # camera position
    lookat = np.array([0.0, 0.5, -2.0])   # point camera is looking at
    vup = np.array([0.0, 1.0, 0.0])       # "up" direction for the camera


    # focal_length is the distance from the camera to the viewport
    # larger = more zoomed in, smaller = more zoomed out
    focal_length = 1.0

    # the viewport is a rectangle in 3D space that we shoot rays through
    # viewport_height=2.0 is arbitrary — it sets the scale of the scene
    viewport_height = 2.0
    viewport_width = viewport_height * aspect_ratio

    # The camera poistion (currently the orgin)
    camera_center = lookfrom

    # 5-10 samples (development), 50-100 samples (final)
    antialising_samples = 16 # number of rays to shoot per pixel for anti-aliasing

    # 15 standard
    depth_limit = 8 # maximum recursion depth for ray bounces (to prevent infinite recursion)

    save_path = "img/render_final.png"
    hdr_save_path = "hdr/render_final.npy"

    aperture = 0.01  # aperture size for depth of field (0 = pinhole camera, larger  = more blur)
    focus_dist = np.linalg.norm(lookfrom - lookat) # distance from camera to the plane in focus (used for depth of field calculations)

    tile_size = 32    # pixels per tile side — smaller = better load balance, more overhead
    num_workers = 16  # number of worker processes; increase for more parallelism, decrease to save memory
    starmap_chunksize = 4  # tiles per IPC message — higher = less overhead, lower = better load balance