import numpy as np

def _codebook():
    np.random.seed(0)
    sample = np.random.randn(1000000)
    q = (np.arange(16) + 0.5) / 16
    return np.quantile(sample, q)

_CODEBOOK = _codebook()

def nf4_quantize_dequant(x: np.ndarray, blocksize: int = 128):
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)
    codes = np.empty_like(arr, dtype=np.uint8)
    deq = np.empty_like(arr, dtype=np.float64)
    for i in range(0, n, blocksize):
        block = arr[i:i+blocksize]
        m = np.max(np.abs(block))
        if m == 0:
            m = 1.0
        y = block / m
        idx = np.abs(y[:, None] - _CODEBOOK).argmin(axis=1)
        codes[i:i+blocksize] = idx.astype(np.uint8)
        deq[i:i+blocksize] = _CODEBOOK[idx] * m
    return codes.reshape(x.shape), deq.reshape(x.shape)
