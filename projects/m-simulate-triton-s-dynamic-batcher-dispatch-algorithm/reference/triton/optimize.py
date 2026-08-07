from triton.simulate import simulate
from triton.metrics import calculate_metrics

def optimize_delay(arrivals: list[int], max_batch_size: int, preferred_batch_sizes: list[int], delays_to_try: list[int], throughput_floor: float, compute_fn):
    best_delay = None
    best_p99 = float('inf')

    for delay in sorted(delays_to_try):
        batches = simulate(arrivals, max_batch_size, preferred_batch_sizes, delay, compute_fn)
        metrics = calculate_metrics(arrivals, batches, compute_fn)

        if metrics["throughput"] >= throughput_floor:
            if metrics["p99_queue_delay"] < best_p99:
                best_p99 = metrics["p99_queue_delay"]
                best_delay = delay

    return best_delay
