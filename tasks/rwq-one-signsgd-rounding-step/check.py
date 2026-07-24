import numpy as np


def _oracle(W, scale, V, grad, lr, qmin, qmax):
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)

    V_new = np.clip(V - lr * np.sign(grad), -0.5, 0.5)
    W_q = np.clip(np.round(W / scale + V_new), qmin, qmax)
    return V_new, W_q


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_abs = 0.0
    ok = 1.0

    for _ in range(6):
        n = int(rng.integers(20, 60))
        W = rng.normal(scale=1.5, size=n)
        scale = float(rng.uniform(0.1, 0.6))
        V = rng.uniform(-0.5, 0.5, size=n)
        grad = rng.normal(size=n)
        # sprinkle a few exact zeros into grad to exercise sign(0) == 0
        zero_idx = rng.choice(n, size=max(1, n // 8), replace=False)
        grad[zero_idx] = 0.0
        lr = float(rng.uniform(0.01, 0.2))
        qmin, qmax = -8.0, 7.0

        V_exp, Wq_exp = _oracle(W, scale, V, grad, lr, qmin, qmax)

        try:
            V_got, Wq_got = sol.signsgd_round_step(
                W.copy(), scale, V.copy(), grad.copy(), lr, qmin, qmax
            )
            V_got = np.asarray(V_got, dtype=np.float64)
            Wq_got = np.asarray(Wq_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "exact_match": 0.0}

        if V_got.shape != V_exp.shape or Wq_got.shape != Wq_exp.shape:
            return {"max_abs_err": float("inf"), "exact_match": 0.0}

        worst_abs = max(worst_abs, float(np.max(np.abs(V_got - V_exp))))
        if not np.array_equal(Wq_got, Wq_exp):
            ok = 0.0

    return {"max_abs_err": worst_abs, "exact_match": ok}
