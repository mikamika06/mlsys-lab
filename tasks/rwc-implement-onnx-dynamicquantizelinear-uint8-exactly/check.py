import numpy as np


def _oracle(x):
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = 0.0, 255.0
    xmin = min(0.0, float(x.min()))
    xmax = max(0.0, float(x.max()))
    y_scale = (xmax - xmin) / (qmax - qmin)
    intermediate_zp = qmin - xmin / y_scale
    y_zero_point = int(np.clip(np.round(intermediate_zp), qmin, qmax))
    y = np.clip(np.round(x / y_scale) + y_zero_point, qmin, qmax).astype(np.uint8)
    return y, y_scale, y_zero_point


FAIL = {"exact_match": 0.0}


def grade(sol, fx) -> dict:
    x_all = np.asarray(fx["x"], dtype=np.float32)
    lengths = np.asarray(fx["lengths"], dtype=np.int64)

    ok = 1.0
    for i in range(x_all.shape[0]):
        L = int(lengths[i])
        x = x_all[i, :L]
        ref_y, ref_scale, ref_zp = _oracle(x)

        try:
            got = sol.dynamic_quantize_linear(x.copy())
            got_y = np.asarray(got["y"], dtype=np.int64)
            got_scale = float(got["y_scale"])
            got_zp = int(got["y_zero_point"])
        except Exception:
            ok = 0.0
            break

        if got_y.shape != ref_y.shape:
            ok = 0.0
            break
        if not np.array_equal(got_y, ref_y.astype(np.int64)):
            ok = 0.0
            break
        if got_zp != ref_zp:
            ok = 0.0
            break
        if abs(got_scale - ref_scale) > 1e-9 * max(1.0, abs(ref_scale)):
            ok = 0.0
            break

    return {"exact_match": ok}
