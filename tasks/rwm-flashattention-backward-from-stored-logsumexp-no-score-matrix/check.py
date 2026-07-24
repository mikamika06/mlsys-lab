import numpy as np


def _oracle(Q, K, V, dO, lse):
    d = Q.shape[1]
    scale = 1.0 / np.sqrt(d)

    scores = (Q @ K.T) * scale
    P = np.exp(scores - lse[:, None])

    dV = P.T @ dO
    dP = dO @ V.T
    row_sum = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - row_sum)

    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    worst = 0.0

    for n, d, dv in [(3, 4, 2), (5, 3, 4), (8, 6, 3)]:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, dv))
        dO = rng.normal(size=(n, dv))

        scores = (Q @ K.T) / np.sqrt(d)
        lse = np.log(np.exp(scores - np.max(scores, axis=1, keepdims=True)).sum(axis=1)) + np.max(
            scores, axis=1
        )

        ref = _oracle(Q, K, V, dO, lse)

        try:
            got = sol.flash_backward(Q, K, V, dO, lse)
            err = max(
                float(np.max(np.abs(got[0] - ref[0]))),
                float(np.max(np.abs(got[1] - ref[1]))),
                float(np.max(np.abs(got[2] - ref[2]))),
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        worst = max(worst, err)

    return {"max_abs_err": worst}
