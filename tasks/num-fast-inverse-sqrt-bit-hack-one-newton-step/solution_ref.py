import numpy as np

MAGIC = np.uint32(0x5F3759DF)


def rsqrt_raw(x: np.ndarray) -> np.ndarray:
    """Magic-constant approximation of 1/sqrt(x), no refinement."""
    x32 = np.asarray(x, dtype=np.float32)
    i = x32.view(np.uint32)
    j = MAGIC - (i >> np.uint32(1))
    return j.view(np.float32)


def rsqrt_newton(x: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step applied to ``rsqrt_raw(x)``."""
    x32 = np.asarray(x, dtype=np.float32)
    y = rsqrt_raw(x32)
    half = np.float32(0.5)
    three_halves = np.float32(1.5)
    return (y * (three_halves - half * x32 * y * y)).astype(np.float32)
