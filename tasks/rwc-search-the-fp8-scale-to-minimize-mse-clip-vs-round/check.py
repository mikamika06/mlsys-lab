import numpy as np


def _e4m3_values():
    values = []
    for sign in (-1, 1):
        for exp in range(-6, 8):
            for mant in range(8):
                if exp == -6:
                    frac = mant / 8.0
                    v = (2 ** -6) * frac
                else:
                    v = (2 ** exp) * (1.0 + mant / 8.0)
                values.append(sign * v)
    values.extend([-448.0, 448.0])
    return np.unique(np.array(values, dtype=np.float64))


_FP8_VALUES = _e4m3_values()


def _quant_dequant(x, scale):
    y = np.clip(x / scale, -448.0, 448.0)
    flat = y.ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = _FP8_VALUES[np.argmin(np.abs(_FP8_VALUES - v))]
    return (out.reshape(x.shape) * scale).astype(np.float64)


def _oracle(x):
    amax = np.max(np.abs(x))
    s0 = amax / 448.0 if amax != 0 else 1.0
    scales = np.array([s0 * (2.0 ** ((i - 4) / 8.0)) for i in range(9)], dtype=np.float64)
    mses = np.array([
        np.mean((x - _quant_dequant(x, s)) ** 2)
        for s in scales
    ], dtype=np.float64)
    return int(np.argmin(mses)), scales, mses


def grade(sol, fx) -> dict:
    cases = [
        np.array([1.0, 20.0, 100.0], dtype=np.float64),
        np.array([-900.0, -12.5, 0.1, 33.0, 448.0], dtype=np.float64),
        np.array([0.001, 0.003, 0.01, 0.2, 2.0, 70.0], dtype=np.float64),
    ]

    arg_ok = 1.0
    mse_ok = 1.0

    for x in cases:
        try:
            got_i, got_s, got_m = sol.search_fp8_scale(x)
        except Exception:
            return {"argmin_index": 0.0, "mse_curve_error": float("inf")}

        ref_i, ref_s, ref_m = _oracle(x)

        if int(got_i) != ref_i:
            arg_ok = 0.0

        got_m = np.asarray(got_m, dtype=np.float64)
        if got_m.shape != ref_m.shape:
            mse_ok = float("inf")
        else:
            mse_ok = min(
                mse_ok,
                float(np.max(np.abs(got_m - ref_m)))
            )

        if np.asarray(got_s).shape != ref_s.shape:
            mse_ok = float("inf")
        else:
            mse_ok = max(
                mse_ok,
                float(np.max(np.abs(np.asarray(got_s) - ref_s)))
            )

    return {
        "argmin_index": arg_ok,
        "mse_curve_error": mse_ok
    }
