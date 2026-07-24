import numpy as np


def accum_error_growth(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    s_ref = np.cumsum(x, dtype=np.float64)
    s16 = np.zeros(n, dtype=np.float64)
    s32 = np.zeros(n, dtype=np.float64)
    acc16 = np.float16(0.0)
    acc32 = np.float32(0.0)
    for i in range(n):
        acc16 = np.float16(acc16 + np.float16(x[i]))
        acc32 = np.float32(acc32 + np.float32(x[i]))
        s16[i] = float(acc16)
        s32[i] = float(acc32)
    err16 = np.abs(s16 - s_ref)
    err32 = np.abs(s32 - s_ref)
    return err16, err32
