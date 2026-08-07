def calculate_kv_bytes(n_layers, n_kv_heads, head_dim, context_len, dtype_bytes=2):
    raise NotImplementedError


def predict_oom(model_weights_bytes, n_layers, n_kv_heads, head_dim, context_lengths, vram_budget_bytes):
    raise NotImplementedError
