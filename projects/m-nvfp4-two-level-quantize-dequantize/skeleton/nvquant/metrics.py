def compute_mse(x, x_hat):
    """Compute mean squared error between original and reconstructed tensors."""
    raise NotImplementedError


def compute_max_abs_err(x, x_hat):
    """Compute maximum absolute error between original and reconstructed tensors."""
    raise NotImplementedError


def compute_effective_bits_per_param(format_name, payload_bits=4, block_size=16, super_block_size=256):
    """Compute total effective bits per parameter including scaling overhead."""
    raise NotImplementedError
