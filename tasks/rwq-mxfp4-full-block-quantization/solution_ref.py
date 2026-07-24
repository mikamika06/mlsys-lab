import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _snap_e2m1(y):
    abs_y = np.abs(y)
    diffs = np.abs(abs_y[..., None] - _MAG)
    idx = np.argmin(diffs, axis=-1)
    return np.sign(y) * _MAG[idx]


def mxfp4_full_block_quantize(W: np.ndarray) -> dict:
    x = np.asarray(W, dtype=np.float64)
    amax = np.max(np.abs(x), axis=1)
    ratio = np.where(amax > 0, amax, 6.0) / 6.0
    e = np.maximum(0, np.ceil(np.log2(ratio))).astype(np.int64)
    scale = np.power(2.0, e)

    y = x / scale[:, None]
    codes = _snap_e2m1(y)
    dequant = codes * scale[:, None]

    return {"scale": scale, "codes": codes, "dequant": dequant}
