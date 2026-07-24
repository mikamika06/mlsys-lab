def _traffic_bytes(n, elem_bytes, tile):
    """Modeled global-memory bytes moved by a T x T-tiled n x n x n matmul.

    Each output tile is computed by streaming the K dimension in T-sized
    chunks, staging one T x T chunk of A and one T x T chunk of B per K-step
    (each read from global memory exactly once, per output tile). Padding to
    a whole number of tiles (ceil(n / T)) in every dimension is charged in
    full, so a T that leaves a small remainder wastes real bytes.
    """
    num_tiles = -(-n // tile)  # ceil(n / tile)
    tile_elems = tile * tile
    ab_bytes = num_tiles * num_tiles * num_tiles * 2 * tile_elems * elem_bytes
    c_bytes = num_tiles * num_tiles * tile_elems * elem_bytes
    return ab_bytes + c_bytes


def _oracle(n, elem_bytes, tile_sizes):
    traffic = [_traffic_bytes(n, elem_bytes, t) for t in tile_sizes]
    best_idx = min(range(len(traffic)), key=lambda i: traffic[i])
    return traffic, best_idx


def grade(sol, fx) -> dict:
    cases = [
        (100, 4, [8, 10, 12]),
        (100, 4, [16, 20, 25, 32]),
        (257, 2, [16, 32, 64, 128]),
        (48, 8, [4, 6, 8, 12, 16, 24, 48]),
        (33, 4, [4, 5, 8, 11, 16, 33]),
    ]

    ok = 1.0
    for n, elem_bytes, tile_sizes in cases:
        ref_traffic, ref_best = _oracle(n, elem_bytes, tile_sizes)

        try:
            got_traffic, got_best = sol.tiled_matmul_traffic(n, elem_bytes, list(tile_sizes))
        except Exception:
            return {"modeled_mem_access": 0.0}

        try:
            got_traffic = [int(v) for v in got_traffic]
            got_best = int(got_best)
        except Exception:
            return {"modeled_mem_access": 0.0}

        if len(got_traffic) != len(ref_traffic):
            ok = 0.0
            continue
        if got_traffic != ref_traffic:
            ok = 0.0
            continue
        if got_best != ref_best:
            ok = 0.0
            continue

    return {"modeled_mem_access": ok}
