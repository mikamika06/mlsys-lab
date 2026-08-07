def select_execution_path(shape: tuple, is_24_sparse: bool, min_alignment: int = 16, min_m: int = 16) -> str:
    M, N, K = shape
    if not is_24_sparse:
        return "dense_unsupported_pattern"
    if M % min_alignment != 0 or N % min_alignment != 0 or K % min_alignment != 0:
        return "dense_fallback_misaligned"
    if M < min_m:
        return "dense_fallback_small_batch"
    return "sparse_24_tensor_core"


def get_dispatch_info(shape: tuple, is_24_sparse: bool) -> dict:
    path = select_execution_path(shape, is_24_sparse)
    return {
        "path": path,
        "is_sparse_kernel": path == "sparse_24_tensor_core",
        "shape": shape,
    }
