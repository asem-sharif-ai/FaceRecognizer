import numpy as np

def _normalize_l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)   # Scalars ONLY

def _denominator(z: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
