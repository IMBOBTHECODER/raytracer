from numba import njit
import numpy as np

@njit(cache=True)
def lerp(a, b, t):
    # Blend between a and b
    return a + t * (b - a)

@njit(cache=True)
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)