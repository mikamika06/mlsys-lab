import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    shapes = [(10, 16), (100, 64), (5, 3)]
    for shape in shapes:
        try:
            emb = rng.standard_normal(shape).astype(np.float64)
            ref = emb / np.sqrt(shape[1])
            got = sol.normalize_embeddings(emb)
            err = max_abs_err(ref, got)
            if err > max_err:
                max_err = err
        except Exception:
            return {"max_abs_err": float("inf")}
    return {"max_abs_err": max_err}
