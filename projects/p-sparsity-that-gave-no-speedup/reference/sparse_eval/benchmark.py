def simulate_kernel_metrics(shape: tuple, path: str, dtype_bytes: int = 2) -> dict:
    M, N, K = shape
    dense_flops = 2 * M * N * K
    if path == "sparse_24_tensor_core":
        flops = M * N * K
        weight_bytes = int(0.5 * M * K * dtype_bytes + M * K * 0.25)
    else:
        flops = dense_flops
        weight_bytes = M * K * dtype_bytes

    act_bytes = (M * K + M * N + N * K) * dtype_bytes
    total_bytes = weight_bytes + act_bytes
    return {
        "flops": flops,
        "weight_bytes": weight_bytes,
        "total_bytes": total_bytes,
        "path": path,
    }
