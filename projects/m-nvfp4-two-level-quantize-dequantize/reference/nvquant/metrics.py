import numpy as np


def compute_mse(x, x_hat):
    """Compute mean squared error between original and reconstructed tensors."""
    x_arr = np.asarray(x, dtype=np.float32)
    x_hat_arr = np.asarray(x_hat, dtype=np.float32)
    return float(np.mean((x_arr - x_hat_arr) ** 2))


def compute_max_abs_err(x, x_hat):
    """Compute maximum absolute error between original and reconstructed tensors."""
    x_arr = np.asarray(x, dtype=np.float32)
    x_hat_arr = np.asarray(x_hat, dtype=np.float32)
    return float(np.max(np.abs(x_arr - x_hat_arr)))


def compute_effective_bits_per_param(format_name, payload_bits=4, block_size=16, super_block_size=256):
    """Compute total effective bits per parameter including scaling overhead."""
    fmt = str(format_name).lower().strip()
    if fmt == "nvfp4":
        overhead = (8.0 / float(block_size)) + (16.0 / float(super_block_size))
        return float(payload_bits + overhead)
    elif fmt == "mxfp4":
        overhead = 8.0 / 32.0
        return float(payload_bits + overhead)
    else:
        raise ValueError(f"Unknown format: {format_name}")
