def wasted_lane_fraction(grid_dim: int, block_dim: int) -> tuple[int, int, float]:
    """
    Compute the number of warps per block, total warps launched across all blocks,
    and the fraction of wasted lanes due to incomplete final warp.
    The fraction is rounded to six decimal places.

    Parameters
    ----------
    grid_dim : int
        Number of thread blocks in the grid.
    block_dim : int
        Number of threads per block.

    Returns
    -------
    tuple[int, int, float]
        (warps_per_block, total_warps, wasted_lane_fraction)
    """
    warps_per_block = (block_dim + 31) // 32
    total_warps = warps_per_block * grid_dim
    wasted_lanes = 32 * warps_per_block - block_dim
    wasted_fraction = round(wasted_lanes / (32 * warps_per_block), 6)
    return warps_per_block, total_warps, wasted_fraction
