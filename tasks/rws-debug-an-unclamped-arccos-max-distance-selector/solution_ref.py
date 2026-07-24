import numpy as np


def select_min_angle_block(query: np.ndarray, candidates: np.ndarray) -> int:
    q = np.asarray(query, dtype=np.float64)
    C = np.asarray(candidates, dtype=np.float64)
    dots = C @ q
    denom = np.linalg.norm(q) * np.linalg.norm(C, axis=1)
    cos = dots / denom
    cos = np.clip(cos, -1.0, 1.0)
    dist = np.arccos(cos)
    return int(np.argmin(dist))
