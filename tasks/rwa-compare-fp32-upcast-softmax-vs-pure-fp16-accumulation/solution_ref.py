import numpy as np


def softmax_fp32(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    y = y - np.max(y, axis=1, keepdims=True)
    e = np.exp(y, dtype=np.float32)
    return e / np.sum(e, axis=1, keepdims=True, dtype=np.float32)
