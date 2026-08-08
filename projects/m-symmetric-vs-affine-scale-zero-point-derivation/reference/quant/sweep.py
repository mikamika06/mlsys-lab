import numpy as np
from quant.derive import derive_symmetric


def block_size_sweep(weights, block_sizes, bits=4):
    flat = weights.flatten()
    n = len(flat)
    errors = []
    for bs in block_sizes:
        total_err = 0.0
        count = 0
        for i in range(0, n, bs):
            block = flat[i:i + bs]
            scale, zp = derive_symmetric(block, bits=bits)
            qmin = -(1 << (bits - 1))
            qmax = (1 << (bits - 1)) - 1
            quantized = np.clip(np.round(block / scale) + zp, qmin, qmax)
            dequantized = (quantized - zp) * scale
            total_err += float(np.sum((block - dequantized) ** 2))
            count += len(block)
        mse = total_err / count if count > 0 else 0.0
        errors.append((bs, mse))
    return errors
