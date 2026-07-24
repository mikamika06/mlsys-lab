import numpy as np
from mlsys import scorers

def _reference_trajectory(grads, lr=0.01, eps=1e-8, decay_rate=0.9):
    grads = np.asarray(grads, dtype=np.float64)
    T, d = grads.shape
    theta = np.zeros(d, dtype=np.float64)
    v = np.zeros(d, dtype=np.float64)
    traj = [theta.copy()]
    for g in grads:
        v = decay_rate * v + (1 - decay_rate) * g**2
        theta -= lr * g / (np.sqrt(v) + eps)
        traj.append(theta.copy())
    return np.stack(traj, axis=0)

def grade(sol, fx) -> dict:
    # deterministic test cases
    rng = np.random.default_rng(42)
    tests = []
    for _ in range(5):
        T = rng.integers(3, 8)
        d = rng.integers(2, 6)
        grads = rng.standard_normal((T, d))
        lr = rng.uniform(0.001, 0.1)
        eps = rng.uniform(1e-9, 1e-7)
        decay_rate = rng.uniform(0.8, 0.99)
        tests.append((grads, lr, eps, decay_rate))

    ok = True
    for grads, lr, eps, decay_rate in tests:
        try:
            got = sol.rmsprop_trajectory(grads, lr=lr, eps=eps, decay_rate=decay_rate)
            got = np.asarray(got, dtype=np.float64)
        except Exception as e:
            return {"max_abs_err": float("inf")}

        ref = _reference_trajectory(grads, lr=lr, eps=eps, decay_rate=decay_rate)

        if got.shape != ref.shape or got.dtype != np.float64:
            ok = False
            break

        err = scorers.max_abs_err(ref, got)
        if err > 1e-9:
            ok = False
            break

    return {"max_abs_err": 0.0 if ok else float("inf")}
