import numpy as np


def _oracle(Q, K, V, O, LSE, dO):
    n, d = Q.shape
    scale = np.sqrt(float(d))
    scores = Q @ K.T / scale
    P = np.exp(scores - LSE[:, None])

    D = np.sum(dO * O, axis=1)

    dP = dO @ V.T
    dS = P * (dP - D[:, None])

    dQ = dS @ K / scale
    dK = dS.T @ Q / scale
    dV = P.T @ dO
    return dQ, dK, dV


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0

    for n, d, dv, tile in [(7, 4, 3, 2), (16, 8, 5, 4), (5, 3, 6, 8)]:
        Q = rng.normal(size=(n, d)).astype(np.float64)
        K = rng.normal(size=(n, d)).astype(np.float64)
        V = rng.normal(size=(n, dv)).astype(np.float64)
        dO = rng.normal(size=(n, dv)).astype(np.float64)

        scores = Q @ K.T / np.sqrt(float(d))
        LSE = np.log(np.exp(scores - np.max(scores, axis=1, keepdims=True)).sum(axis=1)) + np.max(scores, axis=1)
        P = np.exp(scores - LSE[:, None])
        O = P @ V

        ref = _oracle(Q, K, V, O, LSE, dO)

        try:
            got = sol.flash_backward(Q, K, V, O, LSE, dO, tile)
            err = max(
                float(np.max(np.abs(got[0] - ref[0]))),
                float(np.max(np.abs(got[1] - ref[1]))),
                float(np.max(np.abs(got[2] - ref[2]))),
            )
        except Exception:
            return {"max_abs_err": 1.0}

        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
