import numpy as np


def int8_matmul_per_channel(X, W):
    # TODO: this implementation uses one tensor scale per matrix instead of
    # row scales for X and column scales for W.
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    sx = max(np.max(np.abs(X)) / 127.0, 1.0)
    sw = max(np.max(np.abs(W)) / 127.0, 1.0)

    xq = np.clip(np.rint(X / sx), -127, 127).astype(np.int8)
    wq = np.clip(np.rint(W / sw), -127, 127).astype(np.int8)

    acc = xq.astype(np.int32) @ wq.astype(np.int32)
    return acc.astype(np.float64) * sx * sw
