import numpy as np


def _oracle(W1, b1, W2, b2, X_cal, X, k):
    Hc = X_cal @ W1 + b1
    centered = Hc - np.mean(Hc, axis=0, keepdims=True)
    cov = centered.T @ centered / (Hc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    Q = vecs[:, np.argsort(vals)[::-1]]
    H = X @ W1 + b1
    W2_rot = Q.T @ W2
    return (H @ Q[:, :k]) @ W2_rot[:k, :] + b2


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        (32, 8, 5, 3),
        (40, 10, 4, 5),
        (24, 6, 7, 2),
    ]

    worst = 0.0
    for ncal, hidden, out, k in cases:
        d = hidden + 3
        W1 = rng.normal(size=(d, hidden))
        b1 = rng.normal(size=(hidden,))
        W2 = rng.normal(size=(hidden, out))
        b2 = rng.normal(size=(out,))
        X_cal = rng.normal(size=(ncal, d))
        X = rng.normal(size=(12, d))

        try:
            got = sol.rotate_and_slice(
                W1, b1, W2, b2, X_cal, X, k
            )
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(W1, b1, W2, b2, X_cal, X, k)
        got = np.asarray(got, dtype=np.float64)
        err = float(
            np.linalg.norm(got - ref) /
            (np.linalg.norm(ref) + 1e-12)
        )
        worst = max(worst, err)

    return {"rel_err": worst}
