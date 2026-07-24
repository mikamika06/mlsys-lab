import numpy as np


def dequantize_linear(q: np.ndarray, scale, zero_point, axis: int = 0) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    zp = np.asarray(zero_point, dtype=np.float64)

    if scale.ndim == 0:
        s, z = scale, zp
    else:
        shape = [1] * q.ndim
        shape[axis] = -1
        s = scale.reshape(shape)
        z = zp.reshape(shape)

    return (q - z) * s
