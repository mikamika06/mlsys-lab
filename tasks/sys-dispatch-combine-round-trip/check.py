import numpy as np


def _dense_reference(X, expert_idx, gate_weight, W):
    n, d = X.shape
    Y = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        e = int(expert_idx[i])
        Y[i] = gate_weight[i] * (X[i] @ W[e])
    return Y


def grade(sol, fx) -> dict:
    """
    Builds several seeded random top-1-routing MoE instances (X, expert_idx,
    gate_weight, W), computes the dense reference y_i = g_i * (x_i @ W[e(i)])
    directly with NumPy for every token, and compares it element-wise to the
    submission's dispatch/combine output. Reports the worst-case max abs
    error across all trials.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        n = int(rng.integers(5, 40))
        d = int(rng.integers(3, 12))
        E = int(rng.integers(2, 6))
        X = rng.standard_normal((n, d))
        expert_idx = rng.integers(0, E, size=n)
        gate_weight = rng.uniform(0.1, 1.0, size=n)
        W = rng.standard_normal((E, d, d)) * 0.5

        expected = _dense_reference(X, expert_idx, gate_weight, W)

        try:
            got = sol.moe_dispatch_combine(X.copy(), expert_idx.copy(),
                                            gate_weight.copy(), W.copy())
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": 1e9}

        if got.shape != expected.shape:
            return {"max_abs_err": 1e9}

        worst = max(worst, float(np.max(np.abs(got - expected))))
    return {"max_abs_err": worst}
