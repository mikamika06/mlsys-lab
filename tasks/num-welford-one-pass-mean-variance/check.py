import numpy as np

from mlsys import probe, scorers


def _fixture():
    rng = np.random.default_rng(0)
    n = 500
    return 1e6 + rng.standard_normal(n)


def grade(sol, fx) -> dict:
    x = _fixture()

    mean_ref = float(np.mean(x))
    var_ref = float(np.mean((x - mean_ref) ** 2))

    try:
        mean_got, var_got = sol.welford_mean_var(x.copy())
    except Exception:
        return {"rel_err": float("inf"), "line_events": 0.0}

    try:
        mean_got = float(mean_got)
        var_got = float(var_got)
    except Exception:
        return {"rel_err": float("inf"), "line_events": 0.0}

    if not (np.isfinite(mean_got) and np.isfinite(var_got)):
        return {"rel_err": float("inf"), "line_events": 0.0}

    err = scorers.rel_err(
        np.array([mean_ref, var_ref]), np.array([mean_got, var_got])
    )

    # warm up (import / first-call overhead) before the probed call
    try:
        sol.welford_mean_var(x.copy())
        events = probe.count_line_events(sol.welford_mean_var, x.copy())
    except Exception:
        events = 0

    return {"rel_err": float(err), "line_events": float(events)}
