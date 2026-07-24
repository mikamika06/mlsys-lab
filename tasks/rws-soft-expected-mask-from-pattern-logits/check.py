import numpy as np
from mlsys.scorers import rel_err

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z -= np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((4, 3)), rng.standard_normal((3, 5))),
        (rng.standard_normal((2, 7)), rng.standard_normal((7, 1))),
        (rng.standard_normal((10, 6)), rng.standard_normal((6, 8))),
        (rng.standard_normal((1, 4)), rng.standard_normal((4, 4))),
    ]
    max_rel = 0.0
    for logits, patterns in cases:
        try:
            cand = sol.soft_expected_mask(logits, patterns)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _softmax(logits) @ patterns
        err = rel_err(ref, cand)
        if err > max_rel:
            max_rel = err
    return {"rel_err": max_rel}
