def max_context_size(layers, budget_bytes, k_bits, v_bits):
    low = 1
    high = 131072
    best = 0
    while low <= high:
        mid = (low + high) // 2
        total = 0
        for layer in layers:
            kv_heads = layer["kv_heads"]
            head_dim = layer["head_dim"]
            k_bytes_per_token = (kv_heads * head_dim * k_bits) / 8.0
            v_bytes_per_token = (kv_heads * head_dim * v_bits) / 8.0
            total += int((k_bytes_per_token + v_bytes_per_token) * mid)
        if total <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
