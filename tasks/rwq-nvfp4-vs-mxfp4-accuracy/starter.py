import numpy as np


_CODEBOOK = np.array(
    [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0],
    dtype=np.float64,
)


def _bad_quantize(x, block_size):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        scale = float(np.max(np.abs(block)))
        if scale == 0:
            scale = 1.0
        idx = np.argmin(
            np.abs(block[:, None] / scale - _CODEBOOK[None, :]),
            axis=1,
        )
        out[start:end] = scale * _CODEBOOK[idx]
    return out


def fp4_accuracy_comparison(weight):
    # TODO: this incorrectly uses the same per-block max scaling for both formats
    # and ignores the MXFP4 power-of-two scale restriction.
    weight = np.asarray(weight, dtype=np.float32)
    nv = _bad_quantize(weight, 32)
    mx = _bad_quantize(weight, 32)
    nv_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - nv) ** 2)))
    mx_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - mx) ** 2)))
    return nv_rmse, mx_rmse
