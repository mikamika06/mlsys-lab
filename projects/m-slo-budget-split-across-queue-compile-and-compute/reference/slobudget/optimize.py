def find_optimal_batch(slo_ms, base_compile_ms, per_token_ms, max_batch):
    best = 1
    for b in range(1, max_batch + 1):
        time_cost = base_compile_ms + per_token_ms * b
        if time_cost <= slo_ms:
            best = b
    return float(best)
