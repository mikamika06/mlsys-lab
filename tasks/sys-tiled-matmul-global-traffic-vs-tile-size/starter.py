def tiled_matmul_traffic(n: int, elem_bytes: int, tile_sizes: list[int]) -> tuple[list[int], int]:
    """Modeled global-memory traffic (bytes) of a T x T-tiled n x n x n matmul,
    for every T in `tile_sizes`, plus the index of the T minimizing it.

    See task.md for the exact traffic model (K-streamed tiling with
    ceil(n / T) padding charged in full).
    """
    raise NotImplementedError('your code here')
