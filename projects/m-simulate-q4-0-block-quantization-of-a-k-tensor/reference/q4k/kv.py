import numpy as np
from q4k.block import dequantize_array_q4_0, quantize_array_q4_0


def quantize_k_tensor(k_tensor: np.ndarray) -> dict:
    """Quantize a K tensor in block q4_0 format along the last dimension."""
    k_tensor = np.asarray(k_tensor, dtype=np.float32)
    shape = k_tensor.shape
    head_dim = shape[-1]
    n_blocks = head_dim // 32
    leading = shape[:-1]
    reshaped = k_tensor.reshape(-1, 32)
    scales_flat, packed_flat = quantize_array_q4_0(reshaped)
    scales = scales_flat.reshape(*leading, n_blocks)
    packed = packed_flat.reshape(*leading, n_blocks, 16)
    return {"scales": scales, "packed": packed, "shape": shape}


def dequantize_k_tensor(q_dict: dict) -> np.ndarray:
    """Dequantize a quantized K tensor dict back into float32 array."""
    scales = q_dict["scales"]
    packed = q_dict["packed"]
    shape = q_dict["shape"]
    return dequantize_array_q4_0(scales, packed, shape)


def compute_k_quant_stats(k_tensor: np.ndarray, q_dict: dict) -> dict:
    """Compute error and memory compression statistics for a quantized K tensor."""
    k_tensor = np.asarray(k_tensor, dtype=np.float32)
    dequant = dequantize_k_tensor(q_dict)
    err = np.abs(k_tensor - dequant)
    fp32_bytes = int(k_tensor.nbytes)
    q4_0_bytes = int(q_dict["scales"].nbytes + q_dict["packed"].nbytes)
    return {
        "max_abs_err": float(np.max(err)),
        "mean_abs_err": float(np.mean(err)),
        "fp32_bytes": fp32_bytes,
        "q4_0_bytes": q4_0_bytes,
        "compression_ratio": float(fp32_bytes / q4_0_bytes),
    }
