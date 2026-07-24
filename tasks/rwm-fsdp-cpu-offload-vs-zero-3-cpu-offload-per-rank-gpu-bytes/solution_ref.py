def compare_offload_bytes(P, N, bytes_per_element):
    shard = P // N

    fsdp_gpu_bytes = (shard + shard) * bytes_per_element
    zero3_gpu_bytes = (P + shard) * bytes_per_element

    return {
        "fsdp_gpu_bytes": fsdp_gpu_bytes,
        "zero3_gpu_bytes": zero3_gpu_bytes,
        "difference_bytes": zero3_gpu_bytes - fsdp_gpu_bytes,
    }
