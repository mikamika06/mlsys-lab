import numpy as np


def quantize_absmax(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, float(scale)
