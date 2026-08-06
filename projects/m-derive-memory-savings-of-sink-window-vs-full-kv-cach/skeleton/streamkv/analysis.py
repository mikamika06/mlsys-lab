def compute_kv_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    raise NotImplementedError


def compute_sink_window_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, sink_size: int, window_size: int, bytes_per_elem: int = 2) -> int:
    raise NotImplementedError
