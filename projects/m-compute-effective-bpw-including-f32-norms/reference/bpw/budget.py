from bpw.accounting import compute_effective_bpw


def select_quantization(quants, vram_limit_mb, ctx_len, n_layers, hidden_dim, kv_heads, head_dim):
    vram_bytes = vram_limit_mb * 1024 * 1024
    bytes_per_kv_elem = 2
    kv_cache_bytes = 2 * n_layers * ctx_len * kv_heads * head_dim * bytes_per_kv_elem
    available_for_weights = vram_bytes - kv_cache_bytes
    if available_for_weights <= 0:
        return None
    best_quant = None
    max_bpw = -1.0
    for q_name, tensors in quants.items():
        eff_bpw = compute_effective_bpw(tensors)
        total_weights = sum(t["nelements"] for t in tensors)
        weight_bytes = (total_weights * eff_bpw) / 8.0
        if weight_bytes <= available_for_weights:
            if eff_bpw > max_bpw:
                max_bpw = eff_bpw
                best_quant = q_name
    return best_quant
