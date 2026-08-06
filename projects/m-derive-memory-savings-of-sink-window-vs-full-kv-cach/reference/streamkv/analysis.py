def compute_kv_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_elem


def compute_sink_window_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, sink_size: int, window_size: int, bytes_per_elem: int = 2) -> int:
    active_len = min(seq_len, sink_size + window_size)
    return 2 * active_len * num_layers * num_kv_heads * head_dim * bytes_per_elem
