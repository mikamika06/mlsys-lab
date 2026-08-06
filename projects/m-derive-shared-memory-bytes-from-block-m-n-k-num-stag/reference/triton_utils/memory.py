def compute_shared_memory_bytes(block_m: int, block_n: int, block_k: int, num_stages: int, dtype_bytes: int) -> int:
    lhs_bytes = block_m * block_k * num_stages * dtype_bytes
    rhs_bytes = block_k * block_n * num_stages * dtype_bytes
    return lhs_bytes + rhs_bytes
