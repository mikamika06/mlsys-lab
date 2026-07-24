import numpy as np


def _oracle(W, X, s_x):
    alphas = np.arange(20, dtype=np.float64) / 20.0
    base = W @ X
    losses = []
    for alpha in alphas:
        s = np.power(s_x, alpha)
        scaled = W * s[np.newaxis, :]
        max_abs = np.max(np.abs(scaled), axis=1, keepdims=True)
        scale = max_abs / 127.0
        q = np.round(scaled / scale)
        dequant = q * scale
        restored = dequant * (1.0 / s)[np.newaxis, :]
        out = restored @ X
        losses.append(np.linalg.norm(base - out))
    losses = np.asarray(losses, dtype=np.float64)
    return int(np.argmin(losses)), losses


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0, 0.5], [0.3, 1.5, -1.2]], dtype=np.float64),
            np.array([[1.0, 2.0], [-1.0, 0.5], [0.7, -2.0]], dtype=np.float64),
            np.array([1.5, 0.8, 2.2], dtype=np.float64),
        ),
        (
            np.array([[4.0, -1.0], [2.5, 3.0], [-2.0, 0.25]], dtype=np.float64),
            np.array([[0.2, 1.0, -0.5], [2.0, -1.0, 0.4]], dtype=np.float64),
            np.array([0.7, 1.8], dtype=np.float64),
        ),
        (
            np.array([[0.1, 0.2, -0.3, 0.4]], dtype=np.float64),
            np.array([[3.0], [1.0], [-2.0], [0.5]], dtype=np.float64),
            np.array([1.2, 0.9, 1.6, 2.0], dtype=np.float64),
        ),
    ]

    idx_ok = 1.0
    err = 0.0
    for W, X, s_x in cases:
        ref_idx, ref_losses = _oracle(W, X, s_x)
        try:
            got_idx, got_losses = sol.search_awq_alpha(W, X, s_x)
            got_losses = np.asarray(got_losses, dtype=np.float64)
            got_idx = int(got_idx)
        except Exception:
            idx_ok = 0.0
            err = float("inf")
            break
        if got_idx != ref_idx:
            idx_ok = 0.0
        curve_err = np.linalg.norm(got_losses - ref_losses) / (
            np.linalg.norm(ref_losses) + 1e-12
        )
        err = max(err, float(curve_err))
    return {
        "argmin_index": idx_ok,
        "loss_curve_rel_err": err,
    }
