def tiled_matmul_traffic(n: int, elem_bytes: int, tile_sizes: list[int]) -> tuple[list[int], int]:
    """Modeled global-memory traffic (bytes) of a T x T-tiled n x n x n matmul,
    for every T in `tile_sizes`, plus the index of the T minimizing it.

    See task.md for the exact traffic model (K-streamed tiling with
    ceil(n / T) padding charged in full).
    """
    traffic = []
    for tile in tile_sizes:
        num_tiles = -(-n // tile)  # ceil(n / tile)
        tile_elems = tile * tile
        ab_bytes = num_tiles * num_tiles * num_tiles * 2 * tile_elems * elem_bytes
        c_bytes = num_tiles * num_tiles * tile_elems * elem_bytes
        traffic.append(ab_bytes + c_bytes)

    best_idx = min(range(len(traffic)), key=lambda i: traffic[i])
    return traffic, best_idx
