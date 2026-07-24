import numpy as np

QMIN, QMAX = 0, 15
EPS = 1e-8


def _shrink_lp(x, beta, p):
    ax = np.abs(x)
    if p == 1.0:
        mag = 1.0 / beta
    else:
        mag = (1.0 / beta) * np.power(ax + EPS, p - 1.0)
    return np.sign(x) * np.maximum(ax - mag, 0.0)


def _oracle_step(W, s, z, W_q, lp, beta, qmin, qmax):
    raw = W / s[:, None] + z[:, None]
    r = W_q - raw
    W_e = _shrink_lp(r, beta, lp)
    z_new = np.mean(W_q - W_e - W / s[:, None], axis=1)
    W_q_new = np.clip(np.round(W / s[:, None] + z_new[:, None]), qmin, qmax)
    return W_q_new, z_new


def _make_state(rng, d_out, d_in):
    W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 2.0, size=(1, d_in))
    s = (np.max(W, axis=1) - np.min(W, axis=1)) / QMAX
    s = np.where(s > 0, s, 1.0)
    z = np.clip(np.round(-np.min(W, axis=1) / s), QMIN, QMAX)
    W_q = np.clip(np.round(W / s[:, None] + z[:, None]), QMIN, QMAX)
    return W, s, z, W_q


def grade(sol, fx) -> dict:
    """
    Builds several seeded random HQQ states (W, per-row scale s, per-row
    zero-point z, integer codes W_q) and, for a few (lp, beta) settings,
    runs one half-quadratic iteration (generalized-Lp shrinkage of the
    residual, then group-mean zero-point update, then re-quantization)
    with a NumPy oracle. Compares the submission's returned z (max abs
    error) and W_q (exact) to the oracle's.
    """
    rng = np.random.default_rng(0)
    z_worst = 0.0
    wq_ok = 1.0
    for lp, beta in [(0.7, 10.0), (1.0, 5.0), (0.5, 20.0)]:
        d_out = int(rng.integers(3, 7))
        d_in = int(rng.integers(8, 33))
        W, s, z, W_q = _make_state(rng, d_out, d_in)

        Wq_exp, z_exp = _oracle_step(W, s, z, W_q, lp, beta, QMIN, QMAX)

        try:
            Wq_got, z_got = sol.hqq_half_quadratic_step(
                W.copy(), s.copy(), z.copy(), W_q.copy(), lp, beta, QMIN, QMAX
            )
            Wq_got = np.asarray(Wq_got, dtype=np.float64)
            z_got = np.asarray(z_got, dtype=np.float64)
        except Exception:
            return {"z_max_abs_err": float("inf"), "wq_exact_match": 0.0}

        if z_got.shape != z_exp.shape:
            z_worst = float("inf")
        else:
            z_worst = max(z_worst, float(np.max(np.abs(z_got - z_exp))))

        if Wq_got.shape != Wq_exp.shape or not np.array_equal(Wq_got, Wq_exp):
            wq_ok = 0.0

    return {"z_max_abs_err": z_worst, "wq_exact_match": wq_ok}
