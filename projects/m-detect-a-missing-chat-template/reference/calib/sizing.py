def select_calibration_shape(target_tokens, max_seqlen, min_seqlen, mem_limit_mb, bytes_per_token):
    mem_limit_bytes = mem_limit_mb * 1024 * 1024

    best_candidate = None
    min_compute = float('inf')

    for seqlen in range(min_seqlen, max_seqlen + 1):
        mem_per_sample = seqlen * bytes_per_token
        if mem_per_sample > mem_limit_bytes:
            continue

        N = (target_tokens + seqlen - 1) // seqlen
        compute_cost = N * (seqlen ** 2)

        if compute_cost < min_compute:
            min_compute = compute_cost
            best_candidate = (N, seqlen)

    return best_candidate
