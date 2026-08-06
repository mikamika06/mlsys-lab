import numpy as np

def act_order_perm(H):
    """Return permutation indices to sort columns by descending diagonal of H."""
    raise NotImplementedError

def find_damping(H, start_damp=1e-5, step=10.0, max_iter=10):
    """Find the minimum damping factor that makes Cholesky succeed."""
    raise NotImplementedError

def lazy_batch_update(W, H_inv, errors, block_start, block_size):
    """Apply accumulated errors from a block of columns to the remaining columns."""
    raise NotImplementedError
