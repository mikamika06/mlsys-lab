import numpy as np


def quantize_q4_0(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes an array along its last dimension in blocks of 32 using q4_0."""
    orig_shape = x.shape
    N = orig_shape[-1]
    if N % 32 != 0:
        raise ValueError("Last dimension must be divisible by 32")
    x_flat = x.reshape(-1, N // 32, 32)
    amax = np.max(np.abs(x_flat), axis=-1)
    d = (amax / -8.0).astype(np.float16)
    d_f32 = d.astype(np.float32)
    id_val = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)
    x0 = x_flat * id_val[..., np.newaxis]
    xi = np.clip(np.rint(x0).astype(np.int32), -8, 7)
    u = (xi + 8).astype(np.uint8)
    low = u[..., :16]
    high = u[..., 16:]
    qs = (low | (high << 4)).astype(np.uint8)

    num_blocks = N // 32
    scales_out = d.reshape(orig_shape[:-1] + (num_blocks,))
    qs_out = qs.reshape(orig_shape[:-1] + (num_blocks, 16))
    return scales_out, qs_out


def dequantize_q4_0(scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Dequantizes q4_0 scales and packed nibbles back to float32 array."""
    orig_prefix = scales.shape[:-1]
    num_blocks = scales.shape[-1]
    N = num_blocks * 32
    scales_flat = scales.reshape(-1, num_blocks)
    qs_flat = qs.reshape(-1, num_blocks, 16)

    low = (qs_flat & 0x0F).astype(np.int32) - 8
    high = ((qs_flat >> 4) & 0x0F).astype(np.int32) - 8
    xi = np.concatenate([low, high], axis=-1)

    d = scales_flat.astype(np.float32)[..., np.newaxis]
    block = d * xi
    return block.reshape(orig_prefix + (N,))


def quantized_k_dot(q: np.ndarray, scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Computes dot product between query q and quantized K tensor."""
    k_hat = dequantize_q4_0(scales, qs)
    if q.ndim == 1 and k_hat.ndim == 1:
        return np.dot(q, k_hat)
    if q.ndim == 1 and k_hat.ndim >= 2:
        return np.sum(k_hat * q, axis=-1)
    if q.ndim == 2 and k_hat.ndim == 2:
        return np.matmul(q, k_hat.T)
    if q.ndim == k_hat.ndim:
        return np.sum(q * k_hat, axis=-1)
    return np.matmul(q, np.swapaxes(k_hat, -1, -2))


def k_cache_bytes(shape: tuple[int, ...], dtype_str: str) -> int:
    """Returns total bytes required for a K cache tensor of given shape and format."""
    num_elements = int(np.prod(shape))
    if dtype_str == "fp32":
        return num_elements * 4
    if dtype_str == "fp16":
        return num_elements * 2
    if dtype_str == "q4_0":
        N = shape[-1]
        if N % 32 != 0:
            raise ValueError("Last dimension must be divisible by 32")
        num_outer = num_elements // N
        num_blocks = N // 32
        return num_outer * num_blocks * 18
    raise ValueError(f"Unknown dtype {dtype_str}")
