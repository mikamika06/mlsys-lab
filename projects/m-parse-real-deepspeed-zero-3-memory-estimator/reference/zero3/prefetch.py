def compute_prefetch_schedule(layer_computes, layer_comm_times, prefetch_depth):
    n = len(layer_computes)
    total_stall = 0.0
    ready_time = [0.0] * n

    for i in range(n):
        if i == 0:
            for d in range(min(prefetch_depth + 1, n)):
                if d == 0:
                    ready_time[0] = layer_comm_times[0]
                else:
                    ready_time[d] = ready_time[d - 1] + layer_comm_times[d]
        else:
            max_prefetched = min(i + prefetch_depth, n - 1)
            last_ready = ready_time[max_prefetched - 1] if max_prefetched > 0 else 0.0
            if max_prefetched >= i + 1:
                ready_time[max_prefetched] = max(last_ready, ready_time[i - 1]) + layer_comm_times[max_prefetched]

    curr_time = 0.0
    for i in range(n):
        if curr_time < ready_time[i]:
            total_stall += ready_time[i] - curr_time
            curr_time = ready_time[i]
        curr_time += layer_computes[i]

    total_execution_time = curr_time
    sum_comm = sum(layer_comm_times)
    overlap_ratio = max(0.0, 1.0 - (total_stall / sum_comm)) if sum_comm > 0 else 1.0

    return {
        "total_time": total_execution_time,
        "stall_time": total_stall,
        "overlap_ratio": overlap_ratio,
    }


def find_optimal_prefetch_depth(layer_computes, layer_comm_times, memory_per_layer, memory_limit):
    best_depth = 0
    best_time = float("inf")
    n = len(layer_computes)

    for depth in range(n):
        max_mem = sum(memory_per_layer) + max(
            sum(memory_per_layer[i:min(n, i + 1 + depth)]) for i in range(n)
        )
        if max_mem <= memory_limit:
            sched = compute_prefetch_schedule(layer_computes, layer_comm_times, depth)
            if sched["total_time"] < best_time:
                best_time = sched["total_time"]
                best_depth = depth

    return best_depth
