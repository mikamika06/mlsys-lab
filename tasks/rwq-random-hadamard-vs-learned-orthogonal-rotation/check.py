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


def _oracle(W, X, R):
    Y = W @ X
    H = _hadamard(W.shape[0])
    Y_h = _q4(W @ H) @ _q4(H.T @ X)
    Y_r = _q4(W @ R) @ _q4(R.T @ X)
    return (
        float(np.mean((Y_h - Y) ** 2)),
        float(np.mean((Y_r - Y) ** 2)),
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    W = rng.normal(size=(8, 8))
    X = rng.normal(size=(8, 32))

    raw = rng.normal(size=(8, 8))
    R, _ = np.linalg.qr(raw)

    ref_h, ref_r = _oracle(W, X, R)

    try:
        got_h, got_r = sol.w4a4_rotation_mse(W, X, R)
        got_h = float(got_h)
        got_r = float(got_r)
    except Exception:
        return {"mse": 1.0}

    error = max(abs(got_h - ref_h), abs(got_r - ref_r))
    if got_r > got_h:
        error += got_r - got_h

    return {"mse": float(error)}
