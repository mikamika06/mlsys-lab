import numpy as np


def _oracle_log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    shifted = x - m

    denom = 0.0
    for value in shifted:
        denom += np.exp(value)

    return shifted - np.log(denom)


def grade(sol, fx) -> dict:
    cases = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1000.0, 999.0, 998.0, 997.0], dtype=np.float64),
        np.array([-1000.0, -999.5, -1200.0, -10000.0], dtype=np.float64),
        np.array([0.25, -0.75, 8.5, 8.49, -3.2], dtype=np.float64),
        np.array([10000.0, 9999.0, 9998.0, 9900.0], dtype=np.float64),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = np.asarray(sol.streaming_log_softmax(x), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle_log_softmax(x)

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        if not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
