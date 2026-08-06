import numpy as np

def compute_error_curve(tensor, block_sizes):
    """Compute relative quantization error as a function of block size."""
    errors = []
    flat = tensor.flatten()
    n = len(flat)
    for bs in block_sizes:
        total_err = 0.0
        total_norm = 0.0
        for i in range(0, n, bs):
            block = flat[i:i+bs]
            if len(block) == 0:
                continue
            max_val = np.max(np.abs(block))
            if max_val == 0:
                q_block = np.zeros_like(block)
            else:
                scale = max_val / 127.0
                q_block = np.round(block / scale).clip(-127, 127) * scale
            total_err += np.sum((block - q_block) ** 2)
            total_norm += np.sum(block ** 2)
        rel_err = np.sqrt(total_err / (total_norm + 1e-12))
        errors.append(rel_err)
    return errors
