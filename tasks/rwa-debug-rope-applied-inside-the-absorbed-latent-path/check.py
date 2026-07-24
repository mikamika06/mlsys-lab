import numpy as np


def _rope(x, cos, sin):
    y = np.array(x, dtype=np.float64, copy=True)
    n, h = y.shape
    y = y.reshape(n, h // 2, 2)
    a = y[:, :, 0].copy()
    b = y[:, :, 1].copy()
    y[:, :, 0] = a * cos - b * sin
    y[:, :, 1] = a * sin + b * cos
    return y.reshape(n, h)


def _oracle(z, head, w_latent, w_head, cos, sin):
    latent = np.asarray(z, dtype=np.float64) @ np.asarray(w_latent, dtype=np.float64)
    decoupled = np.asarray(head, dtype=np.float64) @ np.asarray(w_head, dtype=np.float64)
    decoupled = _rope(decoupled, cos, sin)
    return np.concatenate([latent, decoupled], axis=1).astype(np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    z = rng.normal(size=(5, 4))
    head = rng.normal(size=(5, 6))
    w_latent = rng.normal(size=(4, 3))
    w_head = rng.normal(size=(6, 6))
    angles = rng.normal(size=(5, 3))
    cos = np.cos(angles)
    sin = np.sin(angles)

    ref = _oracle(z, head, w_latent, w_head, cos, sin)
    try:
        got = sol.mla_kv_features(z, head, w_latent, w_head, cos, sin)
        got = np.asarray(got, dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
