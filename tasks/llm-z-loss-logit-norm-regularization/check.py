import numpy as np


def _oracle_z_loss(logits, targets, lambda_):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    m = np.max(logits, axis=1)
    lse = m + np.log(np.sum(np.exp(logits - m[:, None]), axis=1))
    ce = -logits[np.arange(logits.shape[0]), targets] + lse
    return ce + lambda_ * (lse ** 2)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]]),
            np.array([1, 2]),
            0.01,
        ),
        (
            np.array([[1000.0, 999.0, 998.0], [-1000.0, -999.0, -998.0]]),
            np.array([0, 2]),
            0.1,
        ),
        (
            np.array([
                [0.2, -0.4, 1.5, 0.0],
                [3.0, 2.0, 1.0, -2.0],
                [-5.0, -4.0, -3.0, -2.0],
            ]),
            np.array([2, 0, 3]),
            0.001,
        ),
    ]

    max_err = 0.0
    for logits, targets, lam in cases:
        try:
            got = np.asarray(sol.z_loss(logits, targets, lam), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle_z_loss(logits, targets, lam)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": max_err}
