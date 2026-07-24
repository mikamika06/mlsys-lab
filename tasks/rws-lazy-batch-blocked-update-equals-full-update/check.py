import numpy as np


def _quantize(a, s):
    mx = np.max(np.abs(a))
    if mx == 0:
        return np.zeros_like(a, dtype=np.float64)
    scale = mx / s
    return np.clip(np.round(a / scale), -s, s) * scale


def _oracle(W, X, s):
    W_work = np.asarray(W, dtype=np.float64).copy()
    h_inv = np.linalg.inv(X @ X.T + 1e-6 * np.eye(X.shape[0]))

    for j in range(W_work.shape[1]):
        old = W_work[:, j].copy()
        q = _quantize(old.reshape(-1, 1), s).reshape(-1)
        err = old - q
        W_work[:, j] = q
        if j + 1 < W_work.shape[1]:
            W_work[:, j + 1:] -= (
                (err / h_inv[j, j])[:, None]
                * h_inv[j, j + 1:][None, :]
            )

    return W_work


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.2, -0.8, 0.4], [0.3, 1.1, -1.4]], dtype=np.float64),
            np.array([[1.0, 0.2], [0.1, 1.1], [0.3, -0.4]], dtype=np.float64),
            4,
            2,
        ),
        (
            np.array(
                [[0.7, -1.1, 0.2, 0.9], [-0.4, 0.8, 1.3, -0.6]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 0.0, 0.2], [0.1, 0.8, -0.3], [0.4, 0.2, 1.2], [0.5, -0.1, 0.7]],
                dtype=np.float64,
            ),
            8,
            3,
        ),
        (
            np.arange(15, dtype=np.float64).reshape(3, 5) / 7.0,
            np.eye(5, dtype=np.float64),
            3,
            4,
        ),
    ]

    worst = 0.0
    for W, X, s, blocksize in cases:
        try:
            got = sol.lazy_batch_update(W.copy(), X.copy(), s, blocksize)
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(W, X, s)
        err = np.linalg.norm(np.asarray(got, dtype=np.float64) - ref)
        err /= np.linalg.norm(ref) + 1e-12
        worst = max(worst, float(err))

    return {"rel_err": worst}
