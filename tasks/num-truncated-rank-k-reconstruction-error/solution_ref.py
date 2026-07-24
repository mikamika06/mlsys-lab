import numpy as np


def truncated_rank_k_error(A: np.ndarray, k: int) -> float:
    _, s, _ = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    return float(s[k])
