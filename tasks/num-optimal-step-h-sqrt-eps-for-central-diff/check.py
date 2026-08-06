import numpy as np


def _central_diff_rel_err(f, fprime, x: float, h: float) -> float:
    approx = (f(x + h) - f(x - h)) / (2.0 * h)
    true = fprime(x)
    return abs(approx - true) / (abs(true) + 1e-300)


def _ref_best_h(f, fprime, x: float, h_grid: np.ndarray):
    errs = np.array([_central_diff_rel_err(f, fprime, x, h) for h in h_grid])
    idx = int(np.argmin(errs))
    return float(h_grid[idx]), float(errs[idx])


def grade(sol, fx) -> dict:
    h_grid = np.logspace(-12.0, -1.0, 46)

    cases = [
        (np.sin, np.cos, 1.3),
        (np.exp, np.exp, 0.7),
        (lambda t: t ** 3 - 2.0 * t, lambda t: 3.0 * t ** 2 - 2.0, 2.0),
        (np.log, lambda t: 1.0 / t, 50.0),
        (lambda t: np.cos(3.0 * t), lambda t: -3.0 * np.sin(3.0 * t), -0.4),
    ]

    worst = 0.0
    for f, fprime, x in cases:
        h_ref, err_ref = _ref_best_h(f, fprime, x, h_grid)

        try:
            h_got = sol.best_h_central_diff(f, fprime, float(x), h_grid.tolist())
        except Exception:
            return {"rel_err": float("inf")}

        try:
            h_got = float(h_got)
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(h_got):
            return {"rel_err": float("inf")}

        # must be an actual element of the grid, not an interpolated value
        if h_got not in h_grid.tolist():
            return {"rel_err": float("inf")}

        err_got = _central_diff_rel_err(f, fprime, x, h_got)
        rel = (err_got - err_ref) / (err_ref + 1e-300)
        worst = max(worst, rel)

    return {"rel_err": worst}
