import numpy as np


class Plane:
    def __init__(self, point: np.ndarray, normal: np.ndarray, material=None):
        self.center = point
        self.normal = normal / np.linalg.norm(normal)
        self.material = material


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, colour: np.ndarray,
                 material=None, metal_fuzz=0.0, emission_intensity=8.0,
                 refraction_index=1.5):
        self.center = center
        self.radius = radius
        self.colour = colour
        self.material = material
        self.metal_fuzz = metal_fuzz
        self.emission_intensity = emission_intensity
        self.refraction_index = refraction_index


class World:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def add_object(self, obj):
        self.objects.append(obj)