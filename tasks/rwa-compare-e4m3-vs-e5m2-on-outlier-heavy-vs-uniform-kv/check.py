import numpy as np


def _e4m3_grid():
    """All finite OCP E4M3 (1-4-3, bias 7) representable magnitudes, signed."""
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue  # reserved NaN encoding
            if exp == 0:
                val = (2 ** -6) * (mant / 8.0)          # subnormal
            else:
                val = (2 ** (exp - 7)) * (1.0 + mant / 8.0)  # normal
            vals.add(val)
            vals.add(-val)
    return np.array(sorted(vals), dtype=np.float64)


def _e5m2_grid():
    """All finite OCP E5M2 (1-5-2, bias 15) representable magnitudes, signed."""
    vals = set()
    for exp in range(31):  # exp=31 reserved for inf/NaN
        for mant in range(4):
            if exp == 0:
                val = (2 ** -14) * (mant / 4.0)          # subnormal
            else:
                val = (2 ** (exp - 15)) * (1.0 + mant / 4.0)  # normal
            vals.add(val)
            vals.add(-val)
    return np.array(sorted(vals), dtype=np.float64)


_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_GRIDS = {"e4m3": _e4m3_grid(), "e5m2": _e5m2_grid()}


def _quant_dequant_raw(x, fmt):
    """Round each element to the nearest representable value of `fmt`,
    saturating (clamping) at the format's finite max magnitude. No rescaling."""
    grid = _GRIDS[fmt]
    fmax = _FP8_MAX[fmt]
    x = np.asarray(x, dtype=np.float64)
    clipped = np.clip(x, -fmax, fmax)
    flat = clipped.ravel()
    diffs = np.abs(flat[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    return grid[idx].reshape(x.shape)


def _oracle_errors(x):
    e4 = _quant_dequant_raw(x, "e4m3")
    e5 = _quant_dequant_raw(x, "e5m2")
    return (
        float(np.max(np.abs(e4 - x))),
        float(np.max(np.abs(e5 - x))),
    )


def _make_uniform_kv(rng, shape):
    """Well-scaled KV activations: no value comes close to either format's max."""
    return rng.standard_normal(shape) * 3.0


def _make_outlier_kv(rng, shape):
    """Mostly small values, plus a handful of raw outliers beyond E4M3's 448 max
    (a well-known real phenomenon in transformer KV/activation tensors)."""
    x = rng.standard_normal(shape) * 3.0
    flat = x.ravel()
    n_out = max(1, flat.size // 400)
    idx = rng.choice(flat.size, size=n_out, replace=False)
    flat[idx] = rng.uniform(600.0, 2000.0, size=n_out) * rng.choice([-1.0, 1.0], size=n_out)
    return flat.reshape(shape)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    shape = (32, 64)
    uniform_kv = _make_uniform_kv(rng, shape)
    outlier_kv = _make_outlier_kv(rng, shape)

    ref_u4, ref_u5 = _oracle_errors(uniform_kv)
    ref_o4, ref_o5 = _oracle_errors(outlier_kv)

    try:
        got_u = sol.fp8_format_errors(uniform_kv.copy())
        got_o = sol.fp8_format_errors(outlier_kv.copy())
        gu4, gu5 = float(got_u[0]), float(got_u[1])
        go4, go5 = float(got_o[0]), float(got_o[1])
    except Exception:
        return {
            "uniform_err_diff": float("inf"),
            "outlier_err_diff": float("inf"),
            "order_uniform": 0.0,
            "order_outlier": 0.0,
        }

    uniform_err_diff = max(abs(gu4 - ref_u4), abs(gu5 - ref_u5))
    outlier_err_diff = max(abs(go4 - ref_o4), abs(go5 - ref_o5))

    return {
        "uniform_err_diff": uniform_err_diff,
        "outlier_err_diff": outlier_err_diff,
        "order_uniform": float(gu4 < gu5),   # E4M3 (more mantissa) wins when well-scaled
        "order_outlier": float(go5 < go4),   # E5M2 (more range) wins under raw outliers
    }
