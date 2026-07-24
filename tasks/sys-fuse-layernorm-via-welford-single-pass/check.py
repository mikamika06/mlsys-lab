import numpy as np

from mlsys import probe


def _two_pass_layer_norm(x, gamma, beta, eps):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def _cases():
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (5, 8), (1, 6), (6, 40), (4, 1)]
    out = []
    for b, d in shapes:
        x = rng.standard_normal((b, d)) * 3.0 + 1.0
        gamma = rng.standard_normal(d)
        beta = rng.standard_normal(d)
        out.append((x, gamma, beta))
    return out


def grade(sol, fx) -> dict:
    eps = 1e-5
    max_err = 0.0
    for x, gamma, beta in _cases():
        ref = _two_pass_layer_norm(x, gamma, beta, eps)
        try:
            got = sol.layer_norm_welford(x.copy(), gamma.copy(), beta.copy(), eps)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "line_events": 0.0}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf"), "line_events": 0.0}

        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    # Probe on a moderately wide row so a genuine per-column Welford loop
    # emits a clearly discriminating number of line events.
    x, gamma, beta = _cases()[3]  # (6, 40)
    try:
        sol.layer_norm_welford(x.copy(), gamma.copy(), beta.copy(), eps)  # warm up
        events = probe.count_line_events(
            sol.layer_norm_welford, x.copy(), gamma.copy(), beta.copy(), eps
        )
    except Exception:
        events = 0

    return {"max_abs_err": max_err, "line_events": float(events)}
