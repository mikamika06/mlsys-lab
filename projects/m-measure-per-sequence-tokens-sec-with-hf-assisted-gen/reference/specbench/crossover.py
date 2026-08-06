def find_crossover_batch(batch_sizes, baseline_throughput, speculative_throughput):
    best_b = batch_sizes[0]
    min_diff = float("inf")
    for b, base, spec in zip(batch_sizes, baseline_throughput, speculative_throughput):
        diff = abs(spec - base)
        if diff < min_diff:
            min_diff = diff
            best_b = b
    return best_b
