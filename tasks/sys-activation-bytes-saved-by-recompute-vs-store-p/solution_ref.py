def activation_bytes_saved(N: int, d: int, bytes_per_element: int) -> tuple[int, int]:
    store_bytes = N * N * bytes_per_element
    recompute_bytes = 3 * N * d * bytes_per_element
    return store_bytes, recompute_bytes
