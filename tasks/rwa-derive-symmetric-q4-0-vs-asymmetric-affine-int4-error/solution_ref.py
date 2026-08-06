import math
import numpy as np


def compare_q4_errors(block: np.ndarray) -> dict:
    x = np.asarray(block, dtype=np.float64)

    amax = 0.0
    for val in x:
        v_abs = math.fabs(val)
        if v_abs > amax:
            amax = v_abs

    s_sym = 2.0 * amax / 15.0

    sum_sq_diff_sym = 0.0
    sum_sq_x = 0.0

    for val in x:
        if s_sym == 0.0:
            q = 0.0
        else:
            div = val / s_sym
            r = round(div)
            if r < -8:
                q = -8.0
            elif r > 7:
                q = 7.0
            else:
                q = float(r)
        recon = s_sym * q
        diff = recon - val
        sum_sq_diff_sym += diff * diff
        sum_sq_x += val * val

    err_sym = math.sqrt(sum_sq_diff_sym) / (math.sqrt(sum_sq_x) + 1e-12)

    xmin = float('inf')
    xmax = float('-inf')
    for val in x:
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val

    s_aff = (xmax - xmin) / 15.0

    if s_aff == 0.0:
        z = 0.0
    else:
        z = float(round(-xmin / s_aff))

    sum_sq_diff_aff = 0.0

    for val in x:
        if s_aff == 0.0:
            q = 0.0
        else:
            div = (val / s_aff) + z
            r = round(div)
            if r < 0:
                q = 0.0
            elif r > 15:
                q = 15.0
            else:
                q = float(r)
        recon = s_aff * (q - z)
        diff = recon - val
        sum_sq_diff_aff += diff * diff

    err_aff = math.sqrt(sum_sq_diff_aff) / (math.sqrt(sum_sq_x) + 1e-12)

    return {
        "q4_0_error": float(err_sym),
        "affine_int4_error": float(err_aff),
        "winner": "q4_0" if err_sym <= err_aff else "affine_int4",
    }
