import numpy as np


def _quantize(a, s):
    mx = np.max(np.abs(a))
    if mx == 0:
        return np.zeros_like(a, dtype=np.float64)
    scale = mx / s
    return np.clip(np.round(a / scale), -s, s) * scale


def lazy_batch_update(W, X, s, blocksize):
    W_work = np.asarray(W, dtype=np.float64).copy()
    h_inv = np.linalg.inv(X @ X.T + 1e-6 * np.eye(X.shape[0]))

    n = W_work.shape[1]

    for start in range(0, n, blocksize):
        end = min(n, start + blocksize)

        for j in range(start, end):
            old = W_work[:, j].copy()
            q = _quantize(old.reshape(-1, 1), s).reshape(-1)
            err = old - q
            W_work[:, j] = q

            if j + 1 < n:
                W_work[:, j + 1:] -= (
                    (err / h_inv[j, j])[:, None]
                    * h_inv[j, j + 1:][None, :]
                )

    return W_work
