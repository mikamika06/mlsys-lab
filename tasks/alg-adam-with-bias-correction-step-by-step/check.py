import numpy as np
from mlsys.scorers import max_abs_err

def _reference_trajectory(params0, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = np.zeros_like(params0)
    v = np.zeros_like(params0)
    traj = [params0.copy()]
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        new_params = traj[-1] - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(new_params)
    return np.stack(traj)

def grade(sol, fx) -> dict:
    # deterministic data
    rng = np.random.default_rng(0)
    d = 5
    T = 10
    params0 = rng.standard_normal(d)
    grads = rng.standard_normal((T, d))
    try:
        candidate = sol.adam_trajectory(params0, grads)
    except Exception:
        return {"max_abs_err": float("inf")}
    ref = _reference_trajectory(params0, grads)
    err = max_abs_err(ref, candidate)
    return {"max_abs_err": err}
