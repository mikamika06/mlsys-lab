import numpy as np

def q4_0_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    if n % 32 != 0:
        raise ValueError("Length of weights must be a multiple of 32")
    codes = np.empty(n, dtype=np.uint8)
    scales = np.empty(n // 32, dtype=np.float16)
    for i in range(0, n, 32):
        max_val = 0.0
        for j in range(32):
            val = abs(float(w[i + j]))
            if val > max_val:
                max_val = val
        d = max_val / 8.0
        if d == 0:
            d = 1e-12
        for j in range(32):
            val = float(w[i + j])
            c = int(round(val / d)) + 8
            if c < 0:
                c = 0
            elif c > 15:
                c = 15
            codes[i + j] = c
        scales[i // 32] = d
    return codes, scales

def q4_0_dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.int16)
    n = len(c)
    if n % 32 != 0:
        raise ValueError("Length of codes must be a multiple of 32")
    w_hat = np.empty(n, dtype=np.float64)
    for i in range(0, n, 32):
        d = float(scales[i // 32])
        for j in range(32):
            w_hat[i + j] = float(c[i + j] - 8) * d
    return w_hat
