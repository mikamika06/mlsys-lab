import numpy as np


def vector_wise_int8_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    sx = np.max(np.abs(X), axis=1) / 127.0
    sw = np.max(np.abs(W), axis=0) / 127.0

    safe_sx = np.where(sx == 0.0, 1.0, sx)
    safe_sw = np.where(sw == 0.0, 1.0, sw)

    Xq = np.clip(np.round(X / safe_sx[:, None]), -127, 127)
    Wq = np.clip(np.round(W / safe_sw[None, :]), -127, 127)

    acc = Xq.astype(np.int64) @ Wq.astype(np.int64)
    Y = acc.astype(np.float64) * np.outer(sx, sw)
    return Y
