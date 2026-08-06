def compute_full_kv_memory(num_layers, num_kv_heads, head_dim, seq_len, batch_size, dtype_bytes=2):
    tokens_stored = seq_len
    elements_per_token = 2 * num_layers * num_kv_heads * head_dim
    return batch_size * tokens_stored * elements_per_token * dtype_bytes


def compute_sink_window_memory(num_layers, num_kv_heads, head_dim, sink_size, window_size, seq_len, batch_size, dtype_bytes=2):
    tokens_stored = min(seq_len, sink_size + window_size)
    elements_per_token = 2 * num_layers * num_kv_heads * head_dim
    return batch_size * tokens_stored * elements_per_token * dtype_bytes


def compute_savings_ratio(full_mem, sink_mem):
    if full_mem == 0:
        return 0.0
    return 1.0 - (sink_mem / full_mem)
