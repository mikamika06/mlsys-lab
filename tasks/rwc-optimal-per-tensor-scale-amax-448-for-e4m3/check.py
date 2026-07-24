import numpy as np


def _e4m3_scalar(v):
    if np.isnan(v):
        return 0.0
    sign = -1.0 if v < 0 else 1.0
    a = abs(float(v))
    if a == 0:
        return 0.0
    if a >= 448:
        return sign * 448.0
    if a < 2 ** -6:
        step = 2 ** -9
        return sign * (round(a / step) * step)
    e = int(np.floor(np.log2(a)))
    step = 2.0 ** (e - 3)
    m = round(a / step)
    if m >= 16:
        e += 1
        m = 8
        step = 2.0 ** (e - 3)
    out = m * step
    if e > 8:
        out = 448.0
    return sign * out


def _oracle(x):
    amax = float(np.max(np.abs(x)))
    scale = amax / 448.0 if amax != 0 else 1.0
    if amax == 0:
        return scale, np.zeros_like(x, dtype=np.float64)
    q = np.vectorize(_e4m3_scalar, otypes=[np.float64])(x / scale)
    return scale, q * scale


def grade(sol, fx) -> dict:
    cases = [
        np.array([1.0, -100.0, 200.0], dtype=np.float32),
        np.array([-448.0, 448.0, 12.5, 0.0], dtype=np.float64),
        np.array([[0.1, 0.5], [2.0, -7.0]], dtype=np.float64),
        np.zeros((3, 4), dtype=np.float32),
    ]
    scale_ok = 1.0
    err = 0.0
    for x in cases:
        try:
            scale, got = sol.quantize_fp8_e4m3_amax(x)
        except Exception:
            return {"scale_abs_err": 1.0, "max_abs_err": 1.0}
        ref_scale, ref = _oracle(x)
        scale_ok = min(scale_ok, abs(float(scale) - ref_scale))
        err = max(err, float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref))))
    return {
        "scale_abs_err": scale_ok,
        "max_abs_err": err,
    }
