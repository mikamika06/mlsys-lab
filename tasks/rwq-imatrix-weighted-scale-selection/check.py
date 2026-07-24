import numpy as np


def _weighted_err(x, w, s, qmin, qmax):
    q = np.clip(np.round(x / s), qmin, qmax)
    xhat = q * s
    return float(np.sum(w * (x - xhat) ** 2))


def _oracle_argmin(x, w, grid, qmin, qmax):
    errs = np.array([_weighted_err(x, w, s, qmin, qmax) for s in grid])
    return int(np.argmin(errs)), errs


def _build_case(rng, n, qmax):
    qmin = -(qmax + 1)
    x = rng.normal(scale=0.12, size=n)
    out_idx = int(rng.integers(0, n))
    x[out_idx] = float(rng.choice([-1.0, 1.0])) * rng.uniform(2.0, 4.0)

    w = np.full(n, 10.0)
    w[out_idx] = 0.02  # imatrix says this outlier channel barely matters

    amax = float(np.max(np.abs(x)))
    grid = np.linspace(amax / (qmax * 6.0), amax / qmax, 12)
    return x, w, grid, qmin, qmax


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    worst_rel = 0.0

    for _ in range(6):
        n = int(rng.integers(6, 12))
        qmax = 7  # 4-bit signed symmetric: codes in [-8, 7]
        x, w, grid, qmin, qmax_ = _build_case(rng, n, qmax)

        exp_idx, exp_errs = _oracle_argmin(x, w, grid, qmin, qmax_)
        plain_idx, _ = _oracle_argmin(x, np.ones(n), grid, qmin, qmax_)
        assert exp_idx != plain_idx, "fixture must diverge from plain-MSE argmin"

        try:
            got_idx = int(sol.imatrix_best_scale(x.copy(), w.copy(), grid.copy(), qmin, qmax_))
        except Exception:
            ok = 0.0
            worst_rel = float("inf")
            continue

        if got_idx != exp_idx:
            ok = 0.0

        if not (0 <= got_idx < len(grid)):
            worst_rel = float("inf")
            continue

        got_err = exp_errs[got_idx]
        exp_err = exp_errs[exp_idx]
        rel = abs(got_err - exp_err) / (abs(exp_err) + 1e-12)
        worst_rel = max(worst_rel, rel)

    return {"argmin_index": ok, "rel_err": worst_rel}
