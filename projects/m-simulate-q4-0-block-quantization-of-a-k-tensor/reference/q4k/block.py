import numpy as np


def quantize_block_q4_0(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Quantize a 32-element float32 1D array into q4_0 scale and packed uint8 nibbles."""
    x_flat = np.asarray(x, dtype=np.float32).ravel()
    amax = float(np.max(np.abs(x_flat)))
    if amax == 0.0:
        return 0.0, np.full(16, 136, dtype=np.uint8)
    d = float(amax / -8.0)
    id_val = 1.0 / d
    v = np.round(x_flat * id_val) + 8.0
    q = np.clip(v, 0, 15).astype(np.uint8)
    q_low = q[:16]
    q_high = q[16:]
    qs = (q_low & 0x0F) | ((q_high & 0x0F) << 4)
    return d, qs.astype(np.uint8)


def dequantize_block_q4_0(d: float, qs: np.ndarray) -> np.ndarray:
    """Dequantize scale d and 16 packed uint8 nibbles into a 32-element float32 array."""
    qs = np.asarray(qs, dtype=np.uint8)
    q_low = qs & 0x0F
    q_high = (qs >> 4) & 0x0F
    q = np.concatenate([q_low, q_high])
    return (d * (q.astype(np.float32) - 8.0)).astype(np.float32)


def quantize_array_q4_0(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize array in contiguous 32-element blocks to q4_0 scales and packed nibbles."""
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    x_flat = x.reshape(-1, 32)
    amax = np.max(np.abs(x_flat), axis=-1)
    scales = np.where(amax == 0.0, 0.0, amax / -8.0).astype(np.float32)
    id_arr = np.where(scales == 0.0, 0.0, 1.0 / scales)
    v = np.round(x_flat * id_arr[:, None]) + 8.0
    v = np.where(scales[:, None] == 0.0, 8.0, v)
    q = np.clip(v, 0, 15).astype(np.uint8)
    q_low = q[:, :16]
    q_high = q[:, 16:]
    packed = (q_low & 0x0F) | ((q_high & 0x0F) << 4)
    return scales, packed.astype(np.uint8)


def dequantize_array_q4_0(scales: np.ndarray, packed: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Dequantize scales and packed uint8 nibbles back into original array shape."""
    scales = np.asarray(scales, dtype=np.float32)
    packed = np.asarray(packed, dtype=np.uint8)
    q_low = packed & 0x0F
    q_high = (packed >> 4) & 0x0F
    q = np.concatenate([q_low, q_high], axis=-1)
    x = scales[..., None] * (q.astype(np.float32) - 8.0)
    return x.reshape(original_shape).astype(np.float32)
