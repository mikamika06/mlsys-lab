import numpy as np

def grade(sol, fx) -> dict:
    prior = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    excess = np.array([-0.1, 0.05, 0.02], dtype=np.float64)
    eta = 0.1
    try:
        got = sol.update_weights(prior, excess, eta)
    except Exception:
        return {"rel_err": float("inf"), "sum_close": 0.0}
    ref = prior * np.exp(eta * excess)
    ref /= ref.sum()
    rel_err = np.linalg.norm(got - ref) / (np.linalg.norm(ref)+1e-12)
    sum_close = abs(np.sum(got)-1.0) <= 1e-12
    return {"rel_err": float(rel_err), "sum_close": float(sum_close)}
