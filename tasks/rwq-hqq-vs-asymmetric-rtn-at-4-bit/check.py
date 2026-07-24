import numpy as np


def _rtn(x):
    x = np.asarray(x, dtype=np.float64)
    qmax = 15.0
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = (xmax - xmin) / qmax
    if scale == 0:
        return x.copy()
    zero = int(np.round(-xmin / scale))
    q = np.clip(np.round(x / scale) + zero, 0, 15)
    return scale * (q - zero)


def _hqq(x):
    x = np.asarray(x, dtype=np.float64)
    qmax = 15.0
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    rtn_scale = (xmax - xmin) / qmax
    if rtn_scale == 0:
        return x.copy()

    best_obj = float("inf")
    best = None
    for scale in np.linspace(0.5 * rtn_scale, 1.5 * rtn_scale, 101):
        for zero in range(-32, 33):
            q = np.clip(np.round(x / scale) + zero, 0, 15)
            xhat = scale * (q - zero)
            obj = float(np.sum(np.abs(x - xhat) ** 0.7))
            if obj < best_obj:
                best_obj = obj
                best = xhat
    return best


def _oracle(x):
    h = _hqq(x)
    r = _rtn(x)
    return float(np.mean((h - x) ** 2)), float(np.mean((r - x) ** 2))


def grade(sol, fx) -> dict:
    x = np.array(
        [
            -1.7,
            -1.1,
            -0.8,
            -0.2,
            0.0,
            0.3,
            0.6,
            1.2,
            1.8,
            15.0,
            -0.4,
            0.9,
        ],
        dtype=np.float64,
    )

    ref_hqq, ref_rtn = _oracle(x)
    try:
        got_hqq, got_rtn = sol.compare_4bit_quantizers(x)
        got_hqq = float(got_hqq)
        got_rtn = float(got_rtn)
    except Exception:
        return {
            "hqq_mse_error": float("inf"),
            "rtn_mse_error": float("inf"),
            "hqq_not_worse": 0.0,
        }

    return {
        "hqq_mse_error": abs(got_hqq - ref_hqq),
        "rtn_mse_error": abs(got_rtn - ref_rtn),
        "hqq_not_worse": float(got_hqq <= got_rtn),
    }
