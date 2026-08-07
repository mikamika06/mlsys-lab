def max_allowable_microbatches(num_stages, stage, memory_limit, bytes_per_act):
    for mb in range(1, 1000):
        fwd_in_flight = min(mb, num_stages - stage + 2)
        peak = max(1, fwd_in_flight)
        if peak * bytes_per_act > memory_limit:
            return max(1, mb - 1)
    return 999
