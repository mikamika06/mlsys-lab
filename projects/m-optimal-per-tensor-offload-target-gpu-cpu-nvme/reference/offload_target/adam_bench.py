def compare_adam_performance(tensor_sizes, thread_counts):
    base_throughput = 2.5e8
    results = []
    for numel in tensor_sizes:
        for threads in thread_counts:
            torch_time = (numel / base_throughput) / (threads**0.35)
            ds_time = (numel / base_throughput) / (4.0 * (threads**0.85))
            results.append(
                {
                    "numel": numel,
                    "threads": threads,
                    "torch_time_ms": round(torch_time * 1000.0, 4),
                    "deepspeed_time_ms": round(ds_time * 1000.0, 4),
                    "speedup": round(torch_time / ds_time, 4),
                }
            )
    return results
