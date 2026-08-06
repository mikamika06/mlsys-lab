def launch_indices(block_dim: int, grid_dim: int) -> list[int]:
    """
    Return a flat list of global thread indices for a 1‑D GPU launch.

    Parameters
    ----------
    block_dim : int
        Number of threads per block.
    grid_dim : int
        Number of blocks in the grid.

    Returns
    -------
    list[int]
        A 1-D integer list of length ``block_dim * grid_dim`` containing
        global indices ``block_idx * block_dim + thread_idx`` for each
        combination of block and thread indices.
    """
    result = []
    for i in range(grid_dim):
        for j in range(block_dim):
            result.append(i * block_dim + j)
    return result
