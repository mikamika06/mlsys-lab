import numpy as np


def _oracle_lse(S):
    S = np.asarray(S, dtype=np.float64)
    row_max = np.max(S, axis=1, keepdims=True)
    return (row_max + np.log(np.sum(np.exp(S - row_max), axis=1, keepdims=True))).ravel()


def _oracle_softmax(S):
    S = np.asarray(S, dtype=np.float64)
    row_max = np.max(S, axis=1, keepdims=True)
    exp_shifted = np.exp(S - row_max)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.5, -1001.0]], dtype=np.float64),
        np.arange(24, dtype=np.float64).reshape(4, 6) / 3.0,
        np.array([[0.1], [2.5], [-3.0]], dtype=np.float64),
    ]

    max_err = 0.0
    for S in cases:
        try:
            lse = np.asarray(sol.emit_lse(S), dtype=np.float64)
            probs = np.exp(S - lse[:, None])
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle_softmax(S)
        err = float(np.max(np.abs(probs - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
