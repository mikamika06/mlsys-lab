import numpy as np


def fp32_dot_sum(a: np.ndarray, b: np.ndarray) -> float:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    acc = np.float32(0.0)
    for x, y in zip(a32, b32):
        acc = np.float32(acc + np.float32(x * y))
    return float(acc)
