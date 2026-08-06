def compute_kv_bytes(num_layers, num_kv_heads, head_dim, num_ctx, num_parallel, dtype_bytes=2):
    raise NotImplementedError

def max_feasible_parallel(num_layers, num_kv_heads, head_dim, num_ctx, dtype_bytes, total_vram_bytes, weights_bytes):
    raise NotImplementedError
