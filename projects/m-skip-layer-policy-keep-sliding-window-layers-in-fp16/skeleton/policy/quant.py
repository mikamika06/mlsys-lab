def assign_kv_dtypes(layers):
    """
    Given a list of layer configs (e.g. [{"index": 0, "kind": "full"}, {"index": 1, "kind": "sliding", "window": 1024}]),
    return a list of dtypes ("float8" for full layers, "float16" for sliding window layers).
    """
    raise NotImplementedError

def compute_kv_bytes(layers, dtypes, seq_len, batch_size, kv_heads, head_dim):
    """
    Compute total bytes for the KV cache across all layers.
    "float32" costs 4 bytes, "float16" costs 2 bytes, "float8" costs 1 byte.
    Full attention layer saves elements equal to `seq_len`.
    Sliding window layer saves elements equal to `min(seq_len, window)`.
    Account for 2 tensors (K and V) per element.
    """
    raise NotImplementedError
