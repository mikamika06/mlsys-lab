import numpy as np


def _softmax_temperature(x, tau):
    x = np.asarray(x, dtype=np.float64)
    scaled = x / tau
    scaled = scaled - np.max(scaled)
    exp_x = np.exp(scaled)
    return exp_x / np.sum(exp_x)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([2.0, 1.0, -0.5, 0.3]),
            np.array([0.2, 1.5, -0.1, 0.4]),
            1e-6,
            1e6,
        ),
        (
            np.array([-3.0, 4.0, 1.0]),
            np.array([0.5, -2.0, 0.7]),
            1e-8,
            1e8,
        ),
        (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([-0.4, 0.8, 0.1, -0.2, 0.0]),
            1e-7,
            1e7,
        ),
    ]

    index_ok = 1.0
    rel_values = []

    for logits, g, tau_small, tau_large in cases:
        try:
            got_index, got_dist = sol.gumbel_temperature_limits(
                logits.copy(), g.copy(), tau_small, tau_large
            )
        except Exception:
            return {"argmax_index": 0.0, "rel_err": float("inf")}

        oracle_index = int(np.argmax(logits + g))
        oracle_dist = _softmax_temperature(logits + g, tau_large)

        if int(got_index) != oracle_index:
            index_ok = 0.0

        try:
            rel_values.append(_rel_err(got_dist, oracle_dist))
        except Exception:
            rel_values.append(float("inf"))

    return {
        "argmax_index": index_ok,
        "rel_err": max(rel_values),
    }
