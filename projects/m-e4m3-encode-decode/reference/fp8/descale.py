import numpy as np
from fp8.e4m3 import E4M3_MAX, decode_e4m3, encode_e4m3


def compute_scale(x: np.ndarray) -> float:
    max_val = float(np.max(np.abs(x)))
    if max_val == 0.0:
        return 1.0
    return E4M3_MAX / max_val


def quantize_and_descale(
    x: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    scaled_x = x * scale
    clamped_x = np.clip(scaled_x, -E4M3_MAX, E4M3_MAX)
    q_bytes = encode_e4m3(clamped_x)
    deq = decode_e4m3(q_bytes)
    reconstructed = deq / scale
    return q_bytes, reconstructed
