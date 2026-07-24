import numpy as np

def q4_0_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # TODO: wrong divisor; should be /8 but using /16
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    if n % 32 != 0:
        raise ValueError("Length of weights must be a multiple of 32")
    codes = np.empty(n, dtype=np.uint8)
    scales = np.empty(n // 32, dtype=np.float16)
    for i in range(0, n, 32):
        block = w[i:i+32]
        d = np.max(np.abs(block)) / 16.0   # incorrect divisor
        if d == 0:
            d = 1e-12
        c = np.clip(np.round(block / d).astype(int) + 8, 0, 15)
        codes[i:i+32] = c.astype(np.uint8)
        scales[i // 32] = d
    return codes, scales

def q4_0_dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # TODO: missing subtraction of 8 from codes
    c = np.asarray(codes, dtype=np.int16)
    n = len(c)
    if n % 32 != 0:
        raise ValueError("Length of codes must be a multiple of 32")
    w_hat = np.empty(n, dtype=np.float64)
    for i in range(0, n, 32):
        block_c = c[i:i+32]   # missing -8
        d = scales[i // 32]
        w_hat[i:i+32] = block_c * d
    return w_hat
===== FILE: gen_fi
