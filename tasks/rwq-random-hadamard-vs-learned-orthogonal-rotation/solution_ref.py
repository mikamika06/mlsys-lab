import numpy as np


def _q4(a):
    a = np.asarray(a, dtype=np.float64)
    scale = np.max(np.abs(a)) / 7.0
    if scale == 0:
        return np.zeros_like(a)
    return np.clip(np.round(a / scale), -8, 7) * scale


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def w4a4_rotation_mse(W, X, R):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    Y = W @ X

    H = _hadamard(W.shape[0])
    Y_h = _q4(W @ H) @ _q4(H.T @ X)

    Y_r = _q4(W @ R) @ _q4(R.T @ X)

    return (
        float(np.mean((Y_h - Y) ** 2)),
        float(np.mean((Y_r - Y) ** 2)),
    )
