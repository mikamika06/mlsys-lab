import numpy as np


def _oracle(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    losses = coeffs[:, 0] + coeffs[:, 1] + coeffs[:, 2]
    uniform = float(np.mean(losses))
    z = -losses
    z = z - np.max(z)
    weights = np.exp(z)
    weights = weights / np.sum(weights)
    reweighted = float(np.sum(weights * losses))
    reduction = float(uniform - reweighted)
    return uniform, reweighted, reduction


def _rel_err(a, b):
    return float(abs(a - b) / (abs(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.0, 0.5, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[0.2, 0.1, 0.4], [2.0, -0.5, 0.2], [1.0, 1.0, 1.0]]),
        np.array([
            [5.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [3.0, 1.0, 0.5],
            [0.0, 0.0, 2.0],
        ]),
        np.array([[10.0, -2.0, 0.5], [11.0, -1.0, 0.1]]),
    ]

    scores = {
        "uniform_rel_err": 0.0,
        "reweighted_rel_err": 0.0,
        "reduction_rel_err": 0.0,
    }

    try:
        for coeffs in cases:
            ref = _oracle(coeffs)
            got = sol.compare_sampling(coeffs)
            if len(got) != 3:
                return {k: 1.0 for k in scores}
            scores["uniform_rel_err"] = max(
                scores["uniform_rel_err"], _rel_err(float(got[0]), ref[0])
            )
            scores["reweighted_rel_err"] = max(
                scores["reweighted_rel_err"], _rel_err(float(got[1]), ref[1])
            )
            scores["reduction_rel_err"] = max(
                scores["reduction_rel_err"], _rel_err(float(got[2]), ref[2])
            )
    except Exception:
        return {k: 1.0 for k in scores}

    return scores
