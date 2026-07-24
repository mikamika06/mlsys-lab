import numpy as np


def tinygemm_int4_quantize(W: np.ndarray, group_size: int = 128):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)

    gmin = np.min(Wg, axis=2)
    gmax = np.max(Wg, axis=2)
    scale = (gmax - gmin) / 15.0
    scale_safe = np.where(scale == 0, 1.0, scale)
    zero_point = gmin

    q = np.round((Wg - zero_point[:, :, None]) / scale_safe[:, :, None])
    q = np.clip(q, 0, 15).astype(np.uint8)

    deq = q.astype(np.float64) * scale_safe[:, :, None] + zero_point[:, :, None]

    return (
        q.reshape(rows, cols),
        scale,
        zero_point,
        deq.reshape(rows, cols),
    )
