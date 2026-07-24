import numpy as np

_GRID = np.array([90.0, 95.0, 97.0, 99.0, 99.5, 99.9, 100.0])
_QMAX = 7


def _mse_at(x, p, qmax):
    thr = float(np.percentile(np.abs(x), p))
    if thr <= 0.0:
        thr = 1e-8
    clipped = np.clip(x, -thr, thr)
    scale = thr / qmax
    codes = np.clip(np.round(clipped / scale), -qmax, qmax)
    deq = codes * scale
    return float(np.mean((x - deq) ** 2))


def _oracle(x, grid, qmax):
    errs = np.array([_mse_at(x, p, qmax) for p in grid])
    return int(np.argmin(errs)), errs


def _build_case(rng):
    n = int(rng.integers(700, 1200))
    bulk_scale = float(rng.uniform(0.7, 1.5))
    x = rng.normal(scale=bulk_scale, size=n)
    k = 3
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = rng.choice([-1.0, 1.0], size=k) * (5.0 * bulk_scale)
    return x


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    worst_rel = 0.0

    for _ in range(6):
        x = _build_case(rng)
        exp_idx, exp_errs = _oracle(x, _GRID, _QMAX)
        assert 0 < exp_idx < len(_GRID) - 1, "fixture must have an interior optimum"

        try:
            got_idx, got_mse = sol.percentile_clip_best(x.copy(), _GRID.copy(), _QMAX)
            got_idx = int(got_idx)
            got_mse = float(got_mse)
        except Exception:
            ok = 0.0
            worst_rel = float("inf")
            continue

        if got_idx != exp_idx:
            ok = 0.0

        exp_mse = exp_errs[exp_idx]
        rel = abs(got_mse - exp_mse) / (abs(exp_mse) + 1e-12)
        worst_rel = max(worst_rel, rel)

    return {"argmin_index": ok, "rel_err": worst_rel}
