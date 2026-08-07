import numpy as np


def compute_kv_capacity_gain(num_layers, num_heads, head_dim, seq_len, total_gpu_memory_bytes, non_kv_memory_bytes):
    available_bytes = total_gpu_memory_bytes - non_kv_memory_bytes
    if available_bytes <= 0:
        return {"fp16_max_tokens": 0, "fp8_max_tokens": 0, "capacity_gain_ratio": 0.0}

    bytes_per_token_fp16 = 2 * 2 * num_layers * num_heads * head_dim
    bytes_per_token_fp8 = 1 * 2 * num_layers * num_heads * head_dim

    fp16_max_tokens = available_bytes // bytes_per_token_fp16
    fp8_max_tokens = available_bytes // bytes_per_token_fp8

    ratio = float(fp8_max_tokens / fp16_max_tokens) if fp16_max_tokens > 0 else 0.0

    return {
        "fp16_max_tokens": int(fp16_max_tokens),
        "fp8_max_tokens": int(fp8_max_tokens),
        "capacity_gain_ratio": ratio,
    }
