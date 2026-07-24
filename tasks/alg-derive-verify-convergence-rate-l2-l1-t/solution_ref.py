import numpy as np

def predict_error_curve(A: np.ndarray, T: int) -> np.ndarray:
    vals = np.linalg.eigvalsh(A)
    abs_vals = np.abs(vals)
    idx = np.argsort(-abs_vals)
    lam1 = vals[idx[0]]
    if len(vals) > 1:
        lam2 = vals[idx[1]]
    else:
        lam2 = 0.0
    r = abs(lam2 / lam1) if lam1 != 0 else 0.0
    return np.array([r**t for t in range(T)], dtype=np.float64)
