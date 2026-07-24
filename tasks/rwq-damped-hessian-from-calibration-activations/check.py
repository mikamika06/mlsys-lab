import numpy as np

def grade(sol, fx):
    """Grade damped_hessian against a NumPy oracle."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 64).astype(np.float64)
    percent = 0.01

    # --- NumPy oracle ---
    H_ref = 2.0 * X.T @ X
    damp_ref = percent * float(np.mean(np.diag(H_ref)))

    # --- Learner ---
    try:
        H_learner, damp_learner = sol.damped_hessian(X, percent)
    except Exception:
        return {"rel_err_h": 1.0, "rel_err_damp": 1.0}

    H_learner = np.asarray(H_learner, dtype=np.float64)
    damp_learner = float(damp_learner)

    # --- Compute relative errors ---
    denom_h = float(np.linalg.norm(H_ref)) + 1e-12
    rel_err_h = float(np.linalg.norm(H_learner - H_ref) / denom_h)

    denom_d = abs(damp_ref) + 1e-12
    rel_err_damp = abs(damp_learner - damp_ref) / denom_d

    return {
        "rel_err_h": rel_err_h,
        "rel_err_damp": rel_err_damp,
    }
