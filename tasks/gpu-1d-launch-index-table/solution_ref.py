import numpy as np

def launch_indices(block_dim: int, grid_dim: int) -> np.ndarray:
    """
    Return a flat array of global thread indices for a 1‑D GPU launch.

    Parameters
    ----------
    block_dim : int
        Number of threads per block.
    grid_dim : int
        Number of blocks in the grid.

    Returns
    -------
    np.ndarray
        A 1-D integer array of length ``block_dim * grid_dim`` containing
        global indices ``block_idx * block_dim + thread_idx`` for each
        combination of block and thread indices.
    """
    total = block_dim * grid_dim
    return np.arange(total, dtype=np.int64)
