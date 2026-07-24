import numpy as np


def _oracle_migrate(W, X, s):
    scale = np.asarray(s, dtype=np.float64).reshape(1, -1)
    return np.asarray(W, dtype=np.float64) * scale, np.asarray(X, dtype=np.float64) / scale


def _quantize_dequant(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def _quant_error(W, X):
    Wq = _quantize_dequant(W)
    Xq = _quantize_dequant(X)
    return np.linalg.norm(Wq @ Xq.T - W @ X.T) / (np.linalg.norm(W @ X.T) + 1e-12)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []

    for seed in range(3):
        local = np.random.default_rng(seed)
        W = local.normal(0.0, 1.0, size=(8, 4))
        X = local.normal(0.0, 1.0, size=(16, 4))
        X[:, 0] *= 50.0
        s = np.array([5.0, 1.0, 1.0, 1.0])
        cases.append((W, X, s))

    rel_err = 0.0
    ratios = []

    for W, X, s in cases:
        ref_W, ref_X = _oracle_migrate(W, X, s)
        before = _quant_error(W, X)

        try:
            got_W, got_X = sol.smoothquant_migrate(W, X, s)
            got_W = np.asarray(got_W, dtype=np.float64)
            got_X = np.asarray(got_X, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "quant_error_ratio": float("inf")}

        fp_err = np.linalg.norm(got_W @ got_X.T - W @ X.T) / (
            np.linalg.norm(W @ X.T) + 1e-12
        )
        rel_err = max(rel_err, float(fp_err))

        oracle_err = _quant_error(ref_W, ref_X)
        got_err = _quant_error(got_W, got_X)

        if before == 0:
            ratios.append(float("inf"))
        else:
            ratios.append(float(got_err / before))

        if oracle_err >= before:
            ratios.append(float("inf"))

    return {
        "rel_err": rel_err,
        "quant_error_ratio": float(max(ratios)),
    }
