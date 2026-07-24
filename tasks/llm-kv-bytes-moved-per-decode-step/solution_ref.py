def kv_bytes_per_decode(n_layers, n_heads, d_head, ctx_len, dtype_bytes):
    """Return total bytes read from the KV cache during one decode step.

    The KV cache holds K and V tensors for every layer and head across the
    full context. During decode, the attention kernel reads all of it.
    """
    return n_layers * 2 * n_heads * d_head * ctx_len * dtype_bytes
