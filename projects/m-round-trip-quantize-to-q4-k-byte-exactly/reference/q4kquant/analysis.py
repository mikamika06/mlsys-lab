import numpy as np
from q4kquant.quant import dequantize_q4_0, dequantize_q4_k, quantize_q4_0, quantize_q4_k


def locate_dominant_subblock_mse(x: np.ndarray) -> np.ndarray:
    x_flat = np.asarray(x, dtype=np.float32).ravel()
    b = quantize_q4_k(x_flat)
    x_hat = dequantize_q4_k(b, x_flat.shape)

    n_superblocks = len(x_flat) // 256
    err_sq = (x_flat - x_hat) ** 2
    err_subblocks = err_sq.reshape(n_superblocks, 8, 32)
    return np.mean(err_subblocks, axis=2)


def compare_q4k_vs_q40(x: np.ndarray) -> dict:
    x_flat = np.asarray(x, dtype=np.float32).ravel()
    n_elements = len(x_flat)

    b_q4k = quantize_q4_k(x_flat)
    x_q4k = dequantize_q4_k(b_q4k, x_flat.shape)
    q4k_mse = float(np.mean((x_flat - x_q4k) ** 2))

    b_q40 = quantize_q4_0(x_flat)
    x_q40 = dequantize_q4_0(b_q40, x_flat.shape)
    q40_mse = float(np.mean((x_flat - x_q40) ** 2))

    return {
        "q4k_mse": q4k_mse,
        "q40_mse": q40_mse,
        "q4k_bytes": len(b_q4k),
        "q40_bytes": len(b_q40),
        "bpw": float(len(b_q4k) * 8 / n_elements),
    }
