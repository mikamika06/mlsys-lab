import numpy as np


def calculate_kv_bytes(n_layers, n_kv_heads, head_dim, context_len, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * context_len * dtype_bytes


def predict_oom(model_weights_bytes, n_layers, n_kv_heads, head_dim, context_lengths, vram_budget_bytes):
    results = []
    for ctx in context_lengths:
        kv = calculate_kv_bytes(n_layers, n_kv_heads, head_dim, ctx)
        total = model_weights_bytes + kv + 1024 * 1024 * 512
        results.append(total > vram_budget_bytes)
    return results
