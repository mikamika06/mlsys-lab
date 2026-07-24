import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    # deterministic random activations for reproducibility
    rng = np.random.default_rng(42)
    batch_size = 64
    units = 128
    activations = rng.standard_normal((batch_size, units))
    
    try:
        got = sol.score_importance(activations)
    except Exception:
        return {"rel_err": 1.0}
    
    # reference implementation using NumPy
    ref = np.mean(np.abs(activations), axis=0).astype(np.float64)
    
    error = rel_err(ref, got)
    return {"rel_err": float(error)}
