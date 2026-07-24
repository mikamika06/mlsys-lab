import numpy as np


def _oracle(W_O, W_UV, P, c_V):
    W_O = np.asarray(W_O, dtype=np.float64)
    W_UV = np.asarray(W_UV, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    c_V = np.asarray(c_V, dtype=np.float64)

    values = P @ c_V
    up_projected = values @ W_UV.T
    return up_projected @ W_O.T


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    max_err = 0.0

    cases = [
        (4, 6, 3, 5),
        (7, 5, 4, 8),
        (3, 9, 6, 2),
    ]

    for n, m, latent, up in cases:
        W_O = rng.normal(size=(5, up))
        W_UV = rng.normal(size=(up, latent))
        P = rng.random(size=(n, m))
        P = P / np.sum(P, axis=1, keepdims=True)
        c_V = rng.normal(size=(m, latent))

        expected = _oracle(W_O, W_UV, P, c_V)

        try:
            got = sol.absorb_w_uv(W_O, W_UV, P, c_V)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - expected)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
