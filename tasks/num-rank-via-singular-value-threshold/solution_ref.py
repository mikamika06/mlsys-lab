import numpy as np


def svd_rank(A: np.ndarray, tol: float) -> int:
    singular_values = np.linalg.svd(
        np.asarray(A, dtype=np.float64),
        compute_uv=False,
    )
    return int(np.sum(singular_values > tol))
