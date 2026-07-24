def kv_bytes_per_decode(n_layers, n_heads, d_head, ctx_len, dtype_bytes):
    """Return total bytes read from the KV cache during one decode step."""
    raise NotImplementedError("your code here")
