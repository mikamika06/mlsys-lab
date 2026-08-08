import numpy as np
from .codebooks import build_int4_codebook, build_fp4_codebook, build_nf4_codebook
from .quantize import quantize_blockwise, dequantize_blockwise


def compute_error(tensor, codebook, block_size=64):
    q, absmax = quantize_blockwise(tensor, codebook, block_size)
    deq = dequantize_blockwise(q, absmax, codebook, block_size)
    return float(np.mean((tensor - deq) ** 2))


def compare_distributions():
    np.random.seed(42)
    tensors = {
        'normal': np.random.randn(1024).astype(np.float32),
        'uniform': np.random.uniform(-1, 1, 1024).astype(np.float32),
        'laplace': np.random.laplace(0, 1, 1024).astype(np.float32)
    }

    codebooks = {
        'int4': build_int4_codebook(),
        'fp4': build_fp4_codebook(),
        'nf4': build_nf4_codebook()
    }

    results = {}
    for t_name, t in tensors.items():
        results[t_name] = {}
        for c_name, c in codebooks.items():
            results[t_name][c_name] = compute_error(t, c)

    return results
