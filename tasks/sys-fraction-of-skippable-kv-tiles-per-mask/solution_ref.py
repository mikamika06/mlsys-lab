def skippable_kv_tile_fraction(mask: list[list[bool]], tile_size: int) -> float:
    """Fraction of (query-tile, KV-tile) blocks in `mask` that are entirely
    False -- i.e. blocks a tiled/blocked attention kernel could skip
    computing entirely because no (query, key) pair inside them is
    attended to.

    `mask` is an (n, n) list of lists of booleans, True meaning "attention allowed".
    `n` is divisible by `tile_size`.
    """
    n = len(mask)
    nt = n // tile_size
    skipping_count = 0
    total_blocks = nt * nt
    for i in range(nt):
        for j in range(nt):
            has_any = False
            for ii in range(tile_size):
                for jj in range(tile_size):
                    if mask[i * tile_size + ii][j * tile_size + jj]:
                        has_any = True
                        break
                if has_any:
                    break
            if not has_any:
                skipping_count += 1
    return float(skipping_count / total_blocks)
