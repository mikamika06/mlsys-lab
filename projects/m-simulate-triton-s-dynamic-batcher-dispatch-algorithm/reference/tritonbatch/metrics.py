from tritonbatch.simulator import simulate_batcher


def measure_throughput_vs_delay(requests, max_batch_size, delays, preferred_batch_sizes=None):
    results = {}
    for d in delays:
        batches = simulate_batcher(requests, max_batch_size, d, preferred_batch_sizes)
        if not batches:
            throughput = 0.0
        else:
            total_requests = sum(b["size"] for b in batches)
            total_time = batches[-1]["dispatch_time"] - batches[0]["requests"][0]["arrival_time"]
            if total_time <= 0:
                throughput = float(total_requests)
            else:
                throughput = total_requests / (total_time / 1e6)
        results[d] = throughput
    return results
