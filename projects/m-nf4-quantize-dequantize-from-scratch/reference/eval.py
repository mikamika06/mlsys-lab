import numpy as np
from nf4 import build_nf4_codebook, quantize_tensor, dequantize_tensor


def compare_distributions(w_normal, w_uniform, w_laplace, block_size):
    codebooks = {
        "nf4": build_nf4_codebook(),
        "fp4": np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0.0, 0.0, 0.5, 1, 1.5, 2, 3, 4, 6]) / 6.0,
        "int4": np.linspace(-1, 1, 16)
    }

    dists = {
        "normal": w_normal,
        "uniform": w_uniform,
        "laplace": w_laplace
    }

    results = {}
    for d_name, w in dists.items():
        results[d_name] = {}
        for c_name, cb in codebooks.items():
            idx, amax = quantize_tensor(w, cb, block_size)
            w_approx = dequantize_tensor(idx, amax, cb)
            mse = float(np.mean((w - w_approx)**2))
            results[d_name][c_name] = mse

    return results
