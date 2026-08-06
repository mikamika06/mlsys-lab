import numpy as np


def compare_int4_quantizers(W: np.ndarray) -> tuple[float, float, str]:
    W = np.asarray(W, dtype=np.float64)

    lo = W.flat[0]
    hi = W.flat[0]
    for x in W.flat:
        if x < lo:
            lo = x
        if x > hi:
            hi = x

    if hi == lo:
        sum_sq_err = 0.0
        count = 0
        for x in W.flat:
            diff = lo - x
            sum_sq_err += diff * diff
            count += 1
        affine_err = sum_sq_err / count
    else:
        affine_scale = (hi - lo) / 15.0
        affine_zero = round(-lo / affine_scale)
        sum_sq_err = 0.0
        count = 0
        for x in W.flat:
            val = round(x / affine_scale + affine_zero)
            if val < 0:
                q = 0.0
            elif val > 15:
                q = 15.0
            else:
                q = float(val)
            recon = affine_scale * (q - affine_zero)
            diff = recon - x
            sum_sq_err += diff * diff
            count += 1
        affine_err = sum_sq_err / count

    max_abs = 0.0
    for x in W.flat:
        abs_x = x if x >= 0 else -x
        if abs_x > max_abs:
            max_abs = abs_x

    sym_scale = max_abs / 7.0
    if sym_scale == 0:
        sum_sq_err = 0.0
        count = 0
        for x in W.flat:
            diff = 0.0 - x
            sum_sq_err += diff * diff
            count += 1
        symmetric_err = sum_sq_err / count
    else:
        sum_sq_err = 0.0
        count = 0
        for x in W.flat:
            val = round(x / sym_scale)
            if val < -8:
                q = -8.0
            elif val > 7:
                q = 7.0
            else:
                q = float(val)
            recon = sym_scale * q
            diff = recon - x
            sum_sq_err += diff * diff
            count += 1
        symmetric_err = sum_sq_err / count

    best = "affine" if affine_err <= symmetric_err else "symmetric"

    return affine_err, symmetric_err, best
