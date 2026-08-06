import numpy as np


def accum_error_growth(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    err16 = np.zeros(n, dtype=np.float64)
    err32 = np.zeros(n, dtype=np.float64)
    acc_ref = 0.0
    acc16 = np.float16(0.0)
    acc32 = np.float32(0.0)
    for i in range(n):
        val = float(x[i])
        acc_ref = acc_ref + val
        acc16 = np.float16(acc16 + np.float16(val))
        acc32 = np.float32(acc32 + np.float32(val))
        diff16 = float(acc16) - acc_ref
        diff32 = float(acc32) - acc_ref
        err16[i] = diff16 if diff16 >= 0.0 else -diff16
        err32[i] = diff32 if diff32 >= 0.0 else -diff32
    return err16, err32
