def fusion_traffic(n: int, k: int, dtype_bytes: int):
    """
    HBM read+write traffic for a chain of k elementwise ops over n
    elements, run unfused (k separate kernels) vs fused (one kernel).
    """
    bytes_unfused = 2 * k * n * dtype_bytes
    bytes_fused = 2 * n * dtype_bytes
    return bytes_unfused, bytes_fused
