import numpy as np


def _shrink_op(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    if p == 1.0:
        return np.sign(x) * np.maximum(np.abs(x) - 1.0 / beta, 0.0)
    return np.sign(x) * np.maximum(np.abs(x) - (1.0 / beta) * np.abs(x) ** (p - 1.0), 0.0)


def _oracle(W, scale, zero0, qmin, qmax, lp_norm, beta0, kappa, iters):
    W = np.asarray(W, dtype=np.float64)
    zero = float(zero0)
    beta = float(beta0)
    for _ in range(iters):
        W_q = np.clip(np.round(W * scale + zero), qmin, qmax)
        W_r = (W_q - zero) / scale
        W_e = _shrink_op(W - W_r, beta, lp_norm)
        zero = float(np.mean(W_q - (W - W_e) * scale))
        beta *= kappa

    W_q = np.clip(np.round(W * scale + zero), qmin, qmax)
    W_dq = (W_q - zero) / scale
    return W_q, zero, W_dq


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    worst_rel = 0.0

    cases = [
        dict(n=24, nbits=4, lp_norm=0.7, beta0=1.0, kappa=1.5, iters=10),
        dict(n=40, nbits=3, lp_norm=0.9, beta0=2.0, kappa=1.2, iters=15),
        dict(n=16, nbits=4, lp_norm=1.0, beta0=0.5, kappa=2.0, iters=8),
    ]

    for c in cases:
        n, nbits = c["n"], c["nbits"]
        W = rng.normal(loc=0.0, scale=1.0, size=n)
        outlier_idx = rng.integers(0, n)
        W[outlier_idx] *= 6.0  # a heavy outlier, the whole point of HQQ's Lp loss

        qmin, qmax = 0.0, float((1 << nbits) - 1)
        wmin, wmax = float(np.min(W)), float(np.max(W))
        scale = (qmax - qmin) / (wmax - wmin) if wmax > wmin else 1.0
        zero0 = float(np.round(qmin - wmin * scale))

        Wq_exp, z_exp, Wdq_exp = _oracle(
            W, scale, zero0, qmin, qmax, c["lp_norm"], c["beta0"], c["kappa"], c["iters"]
        )

        try:
            Wq_got, z_got, Wdq_got = sol.hqq_optimize(
                W.copy(), scale, zero0, qmin, qmax, c["lp_norm"], c["beta0"], c["kappa"], c["iters"]
            )
            Wq_got = np.asarray(Wq_got, dtype=np.float64)
            Wdq_got = np.asarray(Wdq_got, dtype=np.float64)
        except Exception:
            ok = 0.0
            worst_rel = float("inf")
            continue

        if Wq_got.shape != Wq_exp.shape or not np.array_equal(Wq_got, Wq_exp):
            ok = 0.0

        if Wdq_got.shape != Wdq_exp.shape:
            worst_rel = float("inf")
            continue

        err = float(np.max(np.abs(Wdq_got - Wdq_exp)))
        worst_rel = max(worst_rel, err)

    return {"exact_match": ok, "max_abs_err": worst_rel}
