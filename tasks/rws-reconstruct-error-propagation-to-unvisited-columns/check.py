import numpy as np


def _oracle(W, X, prune_order):
    W_hat = np.asarray(W, dtype=np.float64).copy()
    n = W_hat.shape[1]

    H = np.asarray(X, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T
    H = H + 1e-6 * np.eye(n, dtype=np.float64)

    L = np.linalg.cholesky(H)
    I = np.eye(n, dtype=np.float64)
    Hinv = np.linalg.solve(L.T, np.linalg.solve(L, I))

    states = []
    for pos, q in enumerate(prune_order):
        removed = W_hat[:, q].copy()
        W_hat[:, q] = 0.0
        for j in prune_order[pos + 1:]:
            W_hat[:, j] += removed * (Hinv[q, j] / Hinv[q, q])
        states.append(W_hat.copy())

    return np.stack(states, axis=0)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]),
            [0, 2, 1],
        ),
        (
            np.array([[2.0, -1.0, 3.0, 4.0], [1.0, 5.0, -2.0, 0.0]]),
            np.array([[1.0, 2.0], [0.5, 1.0], [2.0, -1.0], [1.5, 0.5]]),
            [3, 1, 0, 2],
        ),
    ]

    max_err = 0.0
    for W, X, order in cases:
        ref = _oracle(W, X, order)
        try:
            got = np.asarray(
                sol.reconstruct_pruned_weights(W, X, list(order)),
                dtype=np.float64,
            )
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            err = float("inf")
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
