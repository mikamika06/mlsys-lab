"""
Core simulator for KV cache operations.
"""

def append_tokens(cache_seqlens, block_tables, block_size):
    raise NotImplementedError

def decode_bandwidth(cache_seqlens, num_layers, num_kv_heads, head_dim, dtype_bytes):
    raise NotImplementedError
