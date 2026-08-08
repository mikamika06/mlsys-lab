import numpy as np
from nf4.blockwise import dequantize_blockwise, quantize_blockwise
from nf4.codebook import generate_fp4_codebook, generate_int4_codebook, generate_nf4_codebook


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error between original and reconstructed arrays."""
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def evaluate_format_errors(weights: dict, block_size: int = 64) -> dict:
    """Evaluate MSE error for NF4, FP4, and INT4 formats across weight distributions."""
    codebooks = {
        "nf4": generate_nf4_codebook(),
        "fp4": generate_fp4_codebook(),
        "int4": generate_int4_codebook(),
    }

    results = {}
    for dist_name, w in weights.items():
        results[dist_name] = {}
        for fmt, cb in codebooks.items():
            q_idx, scales, orig_shape = quantize_blockwise(w, cb, block_size=block_size)
            rec = dequantize_blockwise(q_idx, scales, cb, orig_shape)
            results[dist_name][fmt] = compute_mse(w, rec)

    return results
