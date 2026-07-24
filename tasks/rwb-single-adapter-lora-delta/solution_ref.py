import numpy as np


def lora_delta_forward(x, base, A, B, scale):
    x = np.asarray(x, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return base + scale * (x @ A) @ B
