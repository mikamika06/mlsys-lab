def select_optimal_batch_size(candidates, profile_fn, memory_cap):
    best_bs = None
    max_throughput = -1.0
    for bs in sorted(list(candidates)):
        throughput, peak_mem = profile_fn(bs)
        if peak_mem <= memory_cap and throughput > max_throughput:
            max_throughput = throughput
            best_bs = bs
    return best_bs, max_throughput
