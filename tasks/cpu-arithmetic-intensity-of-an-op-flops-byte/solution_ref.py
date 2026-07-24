def arithmetic_intensity(m: int, n: int, k: int, elem_bytes: int = 8) -> float:
    """
    Compute the arithmetic intensity (FLOPs per byte moved) for a naive
    matrix multiplication C = A @ B with shapes (m, k), (k, n).
    """
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * elem_bytes
    return flops / bytes_moved
