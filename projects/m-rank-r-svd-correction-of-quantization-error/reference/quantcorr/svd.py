import numpy as np


def compute_quantization_error(w: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    return w.astype(np.float64) - w_q.astype(np.float64)


def svd_correction(error_matrix: np.ndarray, rank: int):
    u, s, vt = np.linalg.svd(error_matrix, full_matrices=False)
    r = min(rank, len(s))
    u_r = u[:, :r]
    s_r = s[:r]
    vt_r = vt[:r, :]
    sqrt_s = np.sqrt(s_r)
    a = u_r * sqrt_s
    b = sqrt_s[:, np.newaxis] * vt_r
    return a, b


def apply_corrected_quantization(w: np.ndarray, w_q: np.ndarray, rank: int):
    err = compute_quantization_error(w, w_q)
    a, b = svd_correction(err, rank)
    approx_err = a @ b
    return w_q.astype(np.float64) + approx_err
