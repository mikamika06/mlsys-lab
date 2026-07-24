import numpy as np


def _oracle(configs):
    configs = np.asarray(configs, dtype=np.float64)
    alpha = configs[:, 0]
    k = configs[:, 1].astype(np.int64)
    c = configs[:, 2]

    accepted = np.empty_like(alpha)
    one_mask = alpha == 1.0
    accepted[one_mask] = k[one_mask] + 1.0

    normal = ~one_mask
    accepted[normal] = (1.0 - np.power(alpha[normal], k[normal] + 1)) / (1.0 - alpha[normal])

    speedup = accepted / (1.0 + k * c)
    return np.stack([accepted, speedup], axis=1)


def grade(sol, fx) -> dict:
    cases = np.array(
        [
            [0.5, 2, 0.2],
            [0.9, 4, 0.1],
            [0.0, 3, 0.5],
            [1.0, 5, 0.25],
            [0.75, 8, 0.05],
        ],
        dtype=np.float64,
    )

    try:
        got = np.asarray(sol.draft_speedup_model(cases), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    ref = _oracle(cases)

    if got.shape != ref.shape:
        return {"rel_err": float("inf")}

    err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
    return {"rel_err": float(err)}
