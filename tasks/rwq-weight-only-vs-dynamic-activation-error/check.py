import numpy as np


def _sym_quant_per_row(A):
    absmax = np.max(np.abs(A), axis=1, keepdims=True)
    absmax = np.where(absmax == 0.0, 1.0, absmax)
    scale = absmax / 127.0
    codes = np.clip(np.round(A / scale), -127, 127)
    return codes * scale


def _oracle(x, W):
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    y_fp = x @ W.T
    W_hat = _sym_quant_per_row(W)

    y_wo = x @ W_hat.T
    mse_wo = float(np.mean((y_fp - y_wo) ** 2))

    x_hat = _sym_quant_per_row(x)
    y_dyn = x_hat @ W_hat.T
    mse_dyn = float(np.mean((y_fp - y_dyn) ** 2))

    return mse_wo, mse_dyn


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_abs = 0.0
    ordering_ok = 1.0

    for _ in range(6):
        b = int(rng.integers(4, 20))
        d_in = int(rng.integers(8, 40))
        d_out = int(rng.integers(4, 20))
        x = rng.normal(size=(b, d_in))
        W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 2.0, size=(d_out, 1))

        exp_wo, exp_dyn = _oracle(x, W)
        assert exp_wo < exp_dyn, "fixture sanity: weight-only must beat dynamic"

        try:
            got_wo, got_dyn = sol.weight_only_vs_dynamic_mse(x.copy(), W.copy())
            got_wo = float(got_wo)
            got_dyn = float(got_dyn)
        except Exception:
            return {"max_abs_err": float("inf"), "ordering_ok": 0.0}

        worst_abs = max(worst_abs, abs(got_wo - exp_wo), abs(got_dyn - exp_dyn))
        if not (got_wo < got_dyn):
            ordering_ok = 0.0

    return {"max_abs_err": worst_abs, "ordering_ok": ordering_ok}
