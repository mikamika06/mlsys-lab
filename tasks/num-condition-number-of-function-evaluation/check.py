import numpy as np


def _oracle(xs):
    xs = np.asarray(xs, dtype=np.float64)
    f = np.log(xs)
    fp = 1.0 / xs
    return np.abs(xs * fp / f)


def grade(sol, fx) -> dict:
    cases = [
        np.array([0.5, 0.75, 2.0, 4.0], dtype=np.float64),
        np.array([1.0001, 1.01, 0.99, 1.1], dtype=np.float64),
        np.array([0.2, 3.5, 10.0, 100.0], dtype=np.float64),
    ]

    ref_parts = []
    got_parts = []
    for x in cases:
        ref_parts.append(_oracle(x))
        try:
            got = np.asarray(sol.log_condition_number(x), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        if got.shape != x.shape:
            return {"rel_err": float("inf")}
        got_parts.append(got)

    ref = np.concatenate(ref_parts)
    got = np.concatenate(got_parts)
    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
    return {"rel_err": err}
