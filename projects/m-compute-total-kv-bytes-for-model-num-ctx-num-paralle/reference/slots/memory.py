def compute_kv_bytes(num_layers, num_kv_heads, head_dim, num_ctx, num_parallel, dtype_bytes=2):
    return 2 * num_layers * num_kv_heads * head_dim * num_ctx * num_parallel * dtype_bytes

def max_feasible_parallel(num_layers, num_kv_heads, head_dim, num_ctx, dtype_bytes, total_vram_bytes, weights_bytes):
    avail = total_vram_bytes - weights_bytes
    if avail <= 0:
        return 0
    bytes_per_slot = compute_kv_bytes(num_layers, num_kv_heads, head_dim, num_ctx, 1, dtype_bytes)
    return int(avail // bytes_per_slot)
