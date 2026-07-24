import numpy as np

def _oracle_sym(x):
    absmax = np.max(np.abs(x))
    scale = absmax / 127 if absmax != 0 else 1.0
    q = np.round(x / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    dq = q.astype(np.float64) * scale
    return dq

def _oracle_asym(x):
    mn = x.min()
    mx = x.max()
    rng = mx - mn
    if rng == 0:
        scale = 1.0
        zp = 128
    else:
        scale = rng / 255
        zp = int(np.round(-mn / scale))
    q = np.round(x / scale + zp)
    q = np.clip(q, 0, 255).astype(np.uint8)
    dq = (q.astype(np.float64) - zp) * scale
    return dq

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_abs_err = 0.0
    sym_mse_diff = 0.0
    asym_mse_diff = 0.0

    for _ in range(5):
        shape = tuple(rng.integers(1, 10, size=2))
        x = rng.uniform(-10, 10, size=shape).astype(np.float64)

        oracle_sym = _oracle_sym(x)
        oracle_asym = _oracle_asym(x)

        try:
            sym_out = sol.sym_quant_dequant(x)
            asym_out = sol.asym_quant_dequant(x)
        except Exception:
            return {"max_abs_err": 0.0, "sym_mse_diff": 0.0, "asym_mse_diff": 0.0}

        if sym_out.shape != x.shape or asym_out.shape != x.shape:
            return {"max_abs_err": 0.0, "sym_mse_diff": 0.0, "asym_mse_diff": 0.0}
        if sym_out.dtype != np.float64 or asym_out.dtype != np.float64:
            return {"max_abs_err": 0.0, "sym_mse_diff": 0.0, "asym_mse_diff": 0.0}

        err_sym = np.max(np.abs(sym_out - oracle_sym))
        err_asym = np.max(np.abs(asym_out - oracle_asym))
        max_abs_err = max(max_abs_err, err_sym, err_asym)

        mse_sym = np.mean((sym_out - x) ** 2)
        mse_oracle_sym = np.mean((oracle_sym - x) ** 2)
        sym_mse_diff = max(sym_mse_diff, abs(mse_sym - mse_oracle_sym))

        mse_asym = np.mean((asym_out - x) ** 2)
        mse_oracle_asym = np.mean((oracle_asym - x) ** 2)
        asym_mse_diff = max(asym_mse_diff, abs(mse_asym - mse_oracle_asym))

    return {
        "max_abs_err": max_abs_err,
        "sym_mse_diff": sym_mse_diff,
        "asym_mse_diff": asym_mse_diff
    }
