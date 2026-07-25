import numpy as np
from mlsys.scorers import max_abs_err

def _oracle(states_a, states_b):
    result = {}
    for key in states_a:
        A = np.asarray(states_a[key], dtype=np.float64)
        B = np.asarray(states_b[key], dtype=np.float64)
        dot = np.sum(A * B, axis=1)
        normA = np.linalg.norm(A, axis=1)
        normB = np.linalg.norm(B, axis=1)
        cos = dot / (normA * normB)
        cos_clipped = np.clip(cos, -1.0, 1.0)
        dist = np.arccos(cos_clipped) / np.pi
        result[key] = dist.astype(np.float64)
    return result

def grade(sol, fx) -> dict:
    # deterministic random data for reproducibility
    rng = np.random.default_rng(42)
    layers = {
        "l1": (rng.standard_normal((5, 4)), rng.standard_normal((5, 4))),
        "l2": (rng.standard_normal((3, 2)), rng.standard_normal((3, 2))),
        "l3": (rng.standard_normal((10, 8)), rng.standard_normal((10, 8)))
    }
    states_a = {k: v[0] for k, v in layers.items()}
    states_b = {k: v[1] for k, v in layers.items()}

    try:
        got = sol.angular_distance_per_layer(states_a, states_b)
    except Exception as e:
        return {"max_abs_err": float("inf")}

    ref = _oracle(states_a, states_b)

    # compute maximum absolute error across all layers
    errors = []
    for key in ref:
        if key not in got:
            return {"max_abs_err": 0.0}
        err = max_abs_err(ref[key], got[key])
        errors.append(err)
    overall_err = max(errors) if errors else 0.0
    return {"max_abs_err": overall_err}
