def optimize_calibration_params(target_tokens, max_seqlen, available_memory_mb):
    best_n = 1
    best_seqlen = 128
    min_cost = float("inf")
    for seqlen in range(128, min(max_seqlen, 4096) + 1, 128):
        n = max(1, target_tokens // seqlen)
        mem_est = (n * seqlen * 4) / (1024 * 1024)
        if mem_est <= available_memory_mb:
            cost = n * seqlen
            if cost < min_cost:
                min_cost = cost
                best_n = n
                best_seqlen = seqlen
    return {"n": best_n, "seqlen": best_seqlen}
