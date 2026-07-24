import numpy as np

FP4_MAX = 6.0


def mxfp4_block_exponent(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)

    amax = np.max(np.abs(xb), axis=1)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(amax / FP4_MAX))
    exp = np.where(amax == 0, 0.0, exp)

    return exp.astype(np.int64)
