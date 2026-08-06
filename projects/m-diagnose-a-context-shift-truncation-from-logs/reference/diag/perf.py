def compare_latencies(sequential_times, batched_times):
    sum_seq = sum(sequential_times)
    sum_batch = sum(batched_times)
    if sum_seq == 0:
        return 0.0
    return sum_batch / sum_seq
