import numpy as np


def _oracle_jsd_beta(p, q, beta):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = beta * q + (1.0 - beta) * p
    kl_q = np.sum(q * np.log(q / m))
    kl_p = np.sum(p * np.log(p / m))
    return (
        beta * kl_q + (1.0 - beta) * kl_p
    ) / (beta * (1.0 - beta))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.8, 0.2], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            0.999999,
        ),
        (
            np.array([0.1, 0.2, 0.7], dtype=np.float64),
            np.array([0.3, 0.4, 0.3], dtype=np.float64),
            0.000001,
        ),
        (
            np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
            np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
            0.99999,
        ),
        (
            np.array([0.05, 0.15, 0.8], dtype=np.float64),
            np.array([0.2, 0.5, 0.3], dtype=np.float64),
            0.00001,
        ),
    ]

    ref = []
    got = []
    for p, q, beta in cases:
        ref.append(_oracle_jsd_beta(p, q, beta))
        try:
            got.append(float(sol.jsd_beta(p, q, beta)))
        except Exception:
            return {"rel_err": float("inf")}

    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
    return {"rel_err": float(err)}
