import numpy as np

_vals = [0.0]
for e in range(1, 16):
    for m in range(8):
        if e == 15 and m == 7:
            continue
        _vals.append(2**(e - 7) * (1 + m / 8))
for m in range(1, 8):
    _vals.append(2**(-6) * (m / 8))
FP8_VALS = np.array(sorted(_vals))

def quantize_to_fp8_vals(x_scaled):
    shape = x_scaled.shape
    x_flat = np.abs(x_scaled).ravel()
    idx = np.searchsorted(FP8_VALS, x_flat)
    idx = np.clip(idx, 1, len(FP8_VALS) - 1)
    left = FP8_VALS[idx - 1]
    right = FP8_VALS[idx]
    closer = np.where((x_flat - left) < (right - x_flat), left, right)
    return (np.sign(x_scaled).ravel() * closer).reshape(shape)
