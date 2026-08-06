"""
Core simulator for KV cache operations.
"""

def append_tokens(cache_seqlens, block_tables, block_size):
    res = []
    for seqlen, table in zip(cache_seqlens, block_tables):
        block_idx = seqlen // block_size
        block_offset = seqlen % block_size
        res.append((table[block_idx], block_offset))
    return res

def decode_bandwidth(cache_seqlens, num_layers, num_kv_heads, head_dim, dtype_bytes):
    read_tokens = sum(cache_seqlens)
    write_tokens = len(cache_seqlens)
    bytes_per_token = 2 * num_kv_heads * head_dim * dtype_bytes
    layer_bytes = (read_tokens + write_tokens) * bytes_per_token
    return layer_bytes * num_layers
