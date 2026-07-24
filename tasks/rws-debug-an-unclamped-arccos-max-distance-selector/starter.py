import numpy as np


def select_min_angle_block(query: np.ndarray, candidates: np.ndarray) -> int:
    # BUG: cosine similarity is fed straight into arccos without clipping
    # to [-1, 1] first, so float64 rounding on near-parallel vectors can
    # produce a cosine a few ULPs over 1.0 -> arccos returns nan.
    # BUG: picks the LARGEST angular distance instead of the smallest, so
    # it selects the candidate pointing farthest from the query.
    q = np.asarray(query, dtype=np.float64)
    C = np.asarray(candidates, dtype=np.float64)
    dots = C @ q
    denom = np.linalg.norm(q) * np.linalg.norm(C, axis=1)
    cos = dots / denom
    dist = np.arccos(cos)
    return int(np.argmax(dist))
