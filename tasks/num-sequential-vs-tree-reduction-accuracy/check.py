import numpy as np


def _sequential_sum(values):
    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + np.float32(value))
    return total


def _oracle(values):
    return np.sum(values.astype(np.float64), dtype=np.float64)


def _rel_err(value, reference):
    return float(abs(np.float64(value) - reference) / (abs(reference) + 1e-12))


def grade(sol, fx) -> dict:
    values = np.concatenate(
        [
            np.array([1e8], dtype=np.float32),
            np.ones(10000, dtype=np.float32),
            np.array([-1e8], dtype=np.float32),
        ]
    )

    reference = _oracle(values)
    sequential_error = _rel_err(_sequential_sum(values), reference)

    try:
        candidate = float(sol.tree_sum(values.tolist()))
    except Exception:
        return {"rel_err": 1.0}

    tree_error = _rel_err(candidate, reference)
    ratio = tree_error / (sequential_error + 1e-12)

    return {"rel_err": float(ratio)}
