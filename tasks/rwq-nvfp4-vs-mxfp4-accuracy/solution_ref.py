import numpy as np


_CODEBOOK = np.array(
    [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0],
    dtype=np.float64,
)


def _quantize(x, block_size, pow2_scale):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        scale = float(np.max(np.abs(block))) / np.max(np.abs(_CODEBOOK))
        if scale == 0:
            q_scale = 1.0
        elif pow2_scale:
            q_scale = 2.0 ** np.ceil(np.log2(scale))
        else:
            q_scale = scale
        idx = np.argmin(
            np.abs(block[:, None] / q_scale - _CODEBOOK[None, :]),
            axis=1,
        )
        out[start:end] = q_scale * _CODEBOOK[idx]
    return out


def fp4_accuracy_comparison(weight):
    weight = np.asarray(weight, dtype=np.float32)
    nv = _quantize(weight, 16, False)
    mx = _quantize(weight, 32, True)
    nv_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - nv) ** 2)))
    mx_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - mx) ** 2)))
    return nv_rmse, mx_rmse
