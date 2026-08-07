def find_max_context(budget_bytes, layer_configs, kv_types):
    low = 1
    high = 1048576
    best = 0
    while low <= high:
        mid = (low + high) // 2
        total = 0
        for cfg in layer_configs:
            k_bytes_per_token = cfg["kv_heads"] * cfg["head_dim"] * kv_types["k"] // 8
            v_bytes_per_token = cfg["kv_heads"] * cfg["head_dim"] * kv_types["v"] // 8
            total += mid * (k_bytes_per_token + v_bytes_per_token)
        if total <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
