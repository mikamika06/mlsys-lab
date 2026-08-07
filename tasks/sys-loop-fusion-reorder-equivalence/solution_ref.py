def fused_tile_pipeline(A: list[float], B: list[float], C: list[float], tile_size: int):
    """
    D = (A + B) * C; E = relu(D) - A; F = sum(E), computed tile-by-tile:
    each contiguous chunk of `tile_size` elements is fully processed
    (add, relu, subtract, partial-sum) before moving to the next chunk.
    """
    n = len(A)
    E = [0.0] * n
    F = 0.0
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        for i in range(start, end):
            d = (A[i] + B[i]) * C[i]
            e = max(d, 0.0) - A[i]
            E[i] = e
            F += e
    return E, F
