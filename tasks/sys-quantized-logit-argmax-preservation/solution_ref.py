import numpy as np


def quantize_classifier_head(X, W, b):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    scale = np.max(np.abs(W), axis=1) / 127.0
    scale = np.where(scale == 0.0, 1.0, scale)

    q = np.clip(np.round(W / scale[:, None]), -127, 127)
    W_int8 = q.astype(np.int64)

    W_deq = W_int8.astype(np.float64) * scale[:, None]
    logits = X @ W_deq.T + b

    return logits, W_int8, scale
