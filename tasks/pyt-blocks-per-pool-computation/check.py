def _ref(alignment, pool_size, pool_header_size, max_block_size):
    """Oracle: derive blocks_per_pool from obmalloc constants via the formula."""
    available = pool_size - pool_header_size
    result = []
    s = alignment
    while s <= max_block_size:
        result.append(available // s)
        s += alignment
    return result

def grade(sol, fx) -> dict:
    cases = [
        # (alignment, pool_size, pool_header_size, max_block_size)
        (8, 4096, 16, 512),      # classic 64-bit CPython
        (4, 4096, 16, 512),      # 32-bit CPython
        (8, 1024, 16, 128),      # small custom page
        (16, 4096, 8, 256),      # larger alignment, smaller header
        (4080, 4096, 16, 4080),  # edge: single size class (alignment == max)
        (256, 4096, 16, 256),    # edge: one class only
        (8, 4096, 16, 500),      # max_block_size not a multiple of alignment
        (8, 4096, 4088, 512),    # tiny usable space after header
        (1, 64, 4, 16),          # alignment=1, small pool
    ]
    ok = 1.0
    for params in cases:
        try:
            got = list(sol.blocks_per_pool(*params))
        except Exception:
            ok = 0.0
            break
        expected = _ref(*params)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
