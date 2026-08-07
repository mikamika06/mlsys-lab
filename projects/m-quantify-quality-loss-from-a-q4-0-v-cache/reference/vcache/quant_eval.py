import numpy as np


def quantize_q4_0(v_matrix, block_size=32):
    """Quantize floating point array to q4_0 blocks."""
    flat = np.asarray(v_matrix, dtype=np.float32).ravel()
    n = len(flat)
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode="constant")

    blocks = flat.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1) / 7.0
    scales[scales == 0] = 1.0

    q = np.round(blocks / scales[:, None])
    q = np.clip(q, -8, 7).astype(np.int8)

    scales_f16 = scales.astype(np.float16)

    q_unsigned = (q + 8).astype(np.uint8)
    packed = (q_unsigned[:, 1::2] << 4) | (q_unsigned[:, 0::2] & 0x0F)

    return {"scales": scales_f16, "quants": packed, "orig_shape": v_matrix.shape, "pad": pad}


def dequantize_q4_0(quantized_data, shape, block_size=32):
    """Dequantize q4_0 blocks back to float32."""
    scales = quantized_data["scales"].astype(np.float32)
    packed = quantized_data["quants"]

    q0 = (packed & 0x0F).astype(np.int8) - 8
    q1 = ((packed >> 4) & 0x0F).astype(np.int8) - 8

    q = np.empty((packed.shape[0], block_size), dtype=np.int8)
    q[:, 0::2] = q0
    q[:, 1::2] = q1

    dequantized = q * scales[:, None]
    flat = dequantized.ravel()

    orig_len = int(np.prod(shape))
    flat = flat[:orig_len]
    return flat.reshape(shape)


def evaluate_v_cache_loss(v_matrix, block_size=32):
    """Evaluate relative reconstruction error for q4_0 V cache."""
    q_data = quantize_q4_0(v_matrix, block_size=block_size)
    dequant = dequantize_q4_0(q_data, v_matrix.shape, block_size=block_size)

    diff = v_matrix.astype(np.float32) - dequant.astype(np.float32)
    norm_diff = np.linalg.norm(diff)
    norm_orig = np.linalg.norm(v_matrix.astype(np.float32))

    if norm_orig == 0:
        return 0.0
    return float(norm_diff / norm_orig)
