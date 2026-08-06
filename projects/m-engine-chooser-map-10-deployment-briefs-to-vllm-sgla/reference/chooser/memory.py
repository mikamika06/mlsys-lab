def estimate_memory(num_params_b, bits_weight, kv_cache_bytes):
    weight_bytes = int(num_params_b * 1e9 * bits_weight / 8.0)
    return weight_bytes + kv_cache_bytes

def compare_gguf_vs_w4a16(num_params_b, kv_cache_bytes):
    gguf_mem = estimate_memory(num_params_b, 4.5, kv_cache_bytes)
    w4a16_mem = estimate_memory(num_params_b, 4.0, kv_cache_bytes) + int(num_params_b * 1e9 * 0.5 / 8.0)
    return {
        "gguf_q4_bytes": gguf_mem,
        "w4a16_bytes": w4a16_mem,
        "diff_bytes": abs(gguf_mem - w4a16_mem)
    }
