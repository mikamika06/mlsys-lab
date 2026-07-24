def model_access_count(n: int, element_size: int) -> tuple[int, int]:
    unfused = 9 * n * element_size
    fused = 5 * n * element_size
    return unfused, fused
