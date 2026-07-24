import numpy as np


def quantized_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def quantize(x):
        scale = np.max(np.abs(x)) / 127.0
        if scale == 0:
            scale = 1.0
        q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
        return q, scale

    qA, sA = quantize(A)
    qB, sB = quantize(B)
    acc = qA.astype(np.int32) @ qB.astype(np.int32)
    return acc.astype(np.float64) * (sA * sB)
