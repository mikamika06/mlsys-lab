def pairwise_memory_access(n: int, d: int, B: int, line_size: int) -> tuple[int, int]:
    """
    Return (naive_loads, tiled_loads) for an n×d matrix of float64 values.
    Uses only integer arithmetic.
    """
    # number of float64 elements per cache line
    L = line_size // 8
    rows_per_line = (d + L - 1) // L

    # Naïve strategy: load both rows for every ordered pair
    naive_loads = n * n * 2 * rows_per_line

    # Determine block sizes
    blocks = []
    full_blocks = n // B
    rem = n % B
    for _ in range(full_blocks):
        blocks.append(B)
    if rem:
        blocks.append(rem)

    # Tiled strategy: load each block pair once
    tiled_loads = 0
    for sz_i in blocks:
        for sz_j in blocks:
            tiled_loads += 2 * rows_per_line * (sz_i + sz_j)

    return naive_loads, tiled_loads
