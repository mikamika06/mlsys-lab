def _oracle(P, N, bytes_per_element):
    shard = P // N
    fsdp_gpu_bytes = (shard + shard) * bytes_per_element
    zero3_gpu_bytes = (P + shard) * bytes_per_element
    return {
        "fsdp_gpu_bytes": fsdp_gpu_bytes,
        "zero3_gpu_bytes": zero3_gpu_bytes,
        "difference_bytes": zero3_gpu_bytes - fsdp_gpu_bytes,
    }


def grade(sol, fx) -> dict:
    cases = [
        (1024, 8, 2),
        (4096, 16, 4),
        (1000000, 4, 2),
        (7776, 9, 8),
        (32768, 32, 16),
        (12000, 12, 1),
    ]
    ok = 1.0
    for P, N, b in cases:
        try:
            got = sol.compare_offload_bytes(P, N, b)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(P, N, b):
            ok = 0.0
            break
    return {"exact_match": ok}
