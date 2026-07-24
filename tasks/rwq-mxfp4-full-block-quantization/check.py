import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _snap_e2m1(y):
    abs_y = np.abs(y)
    diffs = np.abs(abs_y[..., None] - _MAG)
    idx = np.argmin(diffs, axis=-1)
    return np.sign(y) * _MAG[idx]


def _oracle(W):
    x = np.asarray(W, dtype=np.float64)
    amax = np.max(np.abs(x), axis=1)
    ratio = np.where(amax > 0, amax, 6.0) / 6.0
    e = np.maximum(0, np.ceil(np.log2(ratio))).astype(np.int64)
    scale = np.power(2.0, e)

    y = x / scale[:, None]
    codes = _snap_e2m1(y)
    dequant = codes * scale[:, None]
    return scale, codes, dequant


def grade(sol, fx) -> dict:
    W = fx["mx_w"]
    ref_scale, ref_codes, ref_dequant = _oracle(W)

    try:
        got = sol.mxfp4_full_block_quantize(W.copy())
        got_scale = np.asarray(got["scale"], dtype=np.float64)
        got_codes = np.asarray(got["codes"], dtype=np.float64)
        got_dequant = np.asarray(got["dequant"], dtype=np.float64)
    except Exception:
        return {
            "scale_err": float("inf"),
            "code_err": float("inf"),
            "dequant_err": float("inf"),
        }

    if (
        got_scale.shape != ref_scale.shape
        or got_codes.shape != ref_codes.shape
        or got_dequant.shape != ref_dequant.shape
    ):
        return {
            "scale_err": float("inf"),
            "code_err": float("inf"),
            "dequant_err": float("inf"),
        }

    return {
        "scale_err": float(np.max(np.abs(got_scale - ref_scale))),
        "code_err": float(np.max(np.abs(got_codes - ref_codes))),
        "dequant_err": float(np.max(np.abs(got_dequant - ref_dequant))),
    }
