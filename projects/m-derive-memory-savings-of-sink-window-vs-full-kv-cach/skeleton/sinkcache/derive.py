def compute_full_kv_memory(num_layers, num_kv_heads, head_dim, seq_len, batch_size, dtype_bytes=2):
    raise NotImplementedError


def compute_sink_window_memory(num_layers, num_kv_heads, head_dim, sink_size, window_size, seq_len, batch_size, dtype_bytes=2):
    raise NotImplementedError


def compute_savings_ratio(full_mem, sink_mem):
    raise NotImplementedError
