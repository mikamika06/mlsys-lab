import math
import numpy as np


def _e4m3_oracle_scalar(v):
    sign = -1.0 if v < 0 else 1.0
    a = abs(float(v))
    if a == 0:
        return 0.0
    if a >= 448:
        return sign * 448.0
    if a < 2**-6:
        step = 2**-9
        return sign * (round(a / step) * step)
    e = int(math.floor(math.log2(a)))
    step = 2.0 ** (e - 3)
    m = round(a / step)
    if m >= 16:
        e += 1
        m = 8
        step = 2.0 ** (e - 3)
    return sign * min(448.0, m * step)


def oracle_quantize(x_list):
    x = np.asarray(x_list, dtype=np.float64)
    amax = 0.0
    for i in range(x.size):
        val = abs(float(x.flat[i]))
        if val > amax:
            amax = val
    if amax == 0:
        return 1.0, [0.0] * x.size
    scale = amax / 448.0
    out = []
    for i in range(x.size):
        v = float(x.flat[i]) / scale
        q = _e4m3_oracle_scalar(v)
        out.append(float(q * scale))
    return float(scale), out


def grade(sol, fx) -> dict:
    x_in = [1.0, -100.0, 200.0]
    scale_oracle, x_hat_oracle = oracle_quantize(x_in)
    scale_res, x_hat_res = sol.quantize_fp8_e4m3_amax(x_in)

    scale_abs_err = abs(float(scale_res) - float(scale_oracle))

    x_hat_res_arr = np.array(x_hat_res, dtype=np.float64)
    x_hat_oracle_arr = np.array(x_hat_oracle, dtype=np.float64)

    max_abs_err = float(np.max(np.abs(x_hat_res_arr - x_hat_oracle_arr)))

    return {
        "scale_abs_err": scale_abs_err,
        "max_abs_err": max_abs_err,
    }
