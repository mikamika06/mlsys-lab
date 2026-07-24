import numpy as np

def fp4_quant_dequant(x: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize `x` to FP4 e2m1 format in a blockwise manner and return both
    the integer codes (dtype=int8) and the dequantized float64 array.
    """
    flat = x.ravel()
    n = flat.size
    codes = np.empty_like(flat, dtype=np.int8)
    deq = np.empty_like(flat, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = flat[start:end]
        alpha = np.max(np.abs(block))
        s = alpha / 7.0 if alpha != 0 else 1.0
        q = np.round(block / s)
        q = np.clip(q, -8, 7).astype(np.int8)
        codes[start:end] = q
        deq[start:end] = q.astype(np.float64) * s

    return codes.reshape(x.shape), deq.reshape(x.shape)
