import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for shape in [(10,5),(100,20),(50,1)]:
        X = rng.standard_normal(shape).astype(np.float32)
        try:
            out = sol.compute_activation_norms(X)
        except Exception:
            return {"rel_err": float("inf")}
        ref = np.linalg.norm(X.astype(np.float64), axis=0)
        out = np.asarray(out, dtype=np.float64)
        err = rel_err(ref, out)
        if err > max_rel:
            max_rel = err
    return {"rel_err": max_rel}
