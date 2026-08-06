import math

CONFIGS = [
    {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "dtype_bytes": 2},
    {"batch_size": 4, "seq_len": 4096, "num_heads": 32, "head_dim": 128, "dtype_bytes": 2},
    {"batch_size": 8, "seq_len": 8192, "num_heads": 12, "head_dim": 64, "dtype_bytes": 4},
    {"batch_size": 2, "seq_len": 16384, "num_heads": 64, "head_dim": 128, "dtype_bytes": 2},
    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "head_dim": 32, "dtype_bytes": 2},
]

BUDGETS = [
    1024 * 1024 * 64,
    1024 * 1024 * 512,
    1024 * 1024 * 1024 * 4,
    1024 * 1024 * 1024 * 16,
]


def ref_compute_activation_memory(batch_size, seq_len, num_heads, head_dim, mode="lse", dtype_bytes=2):
    base_tensors = 4 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    if mode == "prob":
        extra = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    elif mode == "lse":
        extra = batch_size * num_heads * seq_len * dtype_bytes
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return base_tensors + extra


def ref_max_sequence_length(batch_size, num_heads, head_dim, memory_budget_bytes, mode="lse", dtype_bytes=2):
    if mode == "lse":
        per_token_bytes = batch_size * num_heads * (4 * head_dim + 1) * dtype_bytes
        return memory_budget_bytes // per_token_bytes
    elif mode == "prob":
        a = batch_size * num_heads * dtype_bytes
        b = 4 * batch_size * num_heads * head_dim * dtype_bytes
        c = -memory_budget_bytes
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0
        n = (-b + math.sqrt(disc)) / (2 * a)
        return int(math.floor(n))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ref_max_batch_size(seq_len, num_heads, head_dim, memory_budget_bytes, mode="lse", dtype_bytes=2):
    single_batch_mem = ref_compute_activation_memory(1, seq_len, num_heads, head_dim, mode=mode, dtype_bytes=dtype_bytes)
    return memory_budget_bytes // single_batch_mem
