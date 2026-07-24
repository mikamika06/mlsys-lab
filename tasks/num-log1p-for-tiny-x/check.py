import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    # Generate test cases of tiny values
    rng = np.random.default_rng(0)
    tests = [
        rng.uniform(-1e-12, 1e-12, size=200),
        rng.uniform(-5e-13, 5e-13, size=100),
        np.array([0.0]),
        rng.standard_normal(size=50) * 1e-14,
    ]
    # Compute reference using NumPy's log1p
    refs = [np.log1p(t) for t in tests]
    try:
        outs = [sol.log1p_tiny(t) for t in tests]
    except Exception as e:
        return {"rel_err": float("inf")}
    # Ensure outputs are float64 and same shape
    for out, ref in zip(outs, refs):
        if out.shape != ref.shape or out.dtype != np.float64:
            return {"rel_err": float("inf")}
    # Compute relative error over all tests concatenated
    err = rel_err(np.concatenate(refs), np.concatenate(outs))
    return {"rel_err": err}
