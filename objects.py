from functions import normalize
import numpy as np


class Plane:
    def __init__(self, point: np.ndarray, normal: np.ndarray, colour_multiplier=0.9):
        self.center = point
        self.normal = normalize(normal)
        self.colour_multiplier = colour_multiplier
        self.material = None

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float) -> float:
        denom = np.dot(ray_d, self.normal)
        if abs(denom) < 1e-6:
            return -1
        t = np.dot(self.center - ray_o, self.normal) / denom
        return t if t_min <= t <= t_max else -1

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        return self.normal

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        if (int(np.floor(hit_point[0])) + int(np.floor(hit_point[2]))) % 2 == 0:
            return np.array([0.8, 0.8, 0.8])
        return np.array([0.1, 0.1, 0.1])

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        random_vec = normalize(np.random.randn(3).astype(np.float64))
        if np.dot(random_vec, normal) < 0:
            random_vec = -random_vec
        scatter_direction = normal + random_vec
        if np.linalg.norm(scatter_direction) < 1e-8:
            scatter_direction = normal
        return scatter_direction


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, colour: np.ndarray,
                 material=None, metal_fuzz=0.0, emission_intensity=8.0,
                 refraction_index=1.5, colour_multiplier=0.9):
        self.center = center
        self.radius = radius
        self.colour = colour
        self.material = material
        self.metal_fuzz = metal_fuzz
        self.emission_intensity = emission_intensity
        self.refraction_index = refraction_index
        self.colour_multiplier = colour_multiplier
        self._last_normal = np.array([0.0, 1.0, 0.0])

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float, t_max: float) -> float:
        oc = ray_o - self.center
        a = np.dot(ray_d, ray_d)
        b = 2.0 * np.dot(oc, ray_d)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return -1
        sqrt_d = np.sqrt(discriminant)
        t = (-b - sqrt_d) / (2 * a)
        if t < t_min or t > t_max:
            t = (-b + sqrt_d) / (2 * a)
            if t < t_min or t > t_max:
                return -1
        hit_point = ray_o + t * ray_d
        outward_normal = normalize(hit_point - self.center)
        self._last_normal = outward_normal if np.dot(ray_d, outward_normal) < 0 else -outward_normal
        return t

    def get_normal(self, ray_d: np.ndarray, t: float) -> np.ndarray:
        return self._last_normal

    def object_colour(self, hit_point: np.ndarray) -> np.ndarray:
        if self.material == "emissive":
            return self.colour * self.emission_intensity
        if self.material == "absorbing":
            return np.zeros(3, dtype=np.float64)
        return self.colour

    def reflect(self, ray_d: np.ndarray, normal: np.ndarray):
        if self.material in ("absorbing", "emissive"):
            return None
        if self.material == "metal":
            d = normalize(ray_d)
            reflected = d - 2 * np.dot(d, normal) * normal
            reflected += self.metal_fuzz * normalize(np.random.randn(3))
            return reflected if np.dot(reflected, normal) > 0 else None
        if self.material == "glass":
            d = ray_d
            n = normal if np.dot(d, normal) < 0 else -normal
            eta = 1.0 / self.refraction_index if np.dot(d, normal) < 0 else self.refraction_index
            cos_theta = min(np.dot(-d, n), 1.0)
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))
            r0 = ((1 - eta) / (1 + eta)) ** 2
            reflectance = r0 + (1 - r0) * (1 - cos_theta) ** 5
            if eta * sin_theta > 1.0 or np.random.random() < reflectance:
                return d - 2 * np.dot(d, n) * n
            d_perp = eta * (d + cos_theta * n)
            d_parallel = -np.sqrt(abs(1.0 - np.dot(d_perp, d_perp))) * n
            return d_perp + d_parallel
        random_vec = normalize(np.random.randn(3).astype(np.float64))
        if np.dot(random_vec, normal) < 0:
            random_vec = -random_vec
        scatter_direction = normal + random_vec
        if np.linalg.norm(scatter_direction) < 1e-8:
            scatter_direction = normal
        return scatter_direction


class World:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def add_object(self, obj):
        self.objects.append(obj)

    def hit(self, ray_o: np.ndarray, ray_d: np.ndarray, t_min: float = 0.001, t_max: float = float('inf')) -> tuple:
        closest_t = float('inf')
        hit_obj = None

        for obj in self.objects:
            t = obj.hit(ray_o, ray_d, t_min, t_max)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_obj = obj

        return closest_t, hit_obj