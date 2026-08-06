import numpy as np


def derive_default_num_streams(cpu_cores, batch_size, hint="throughput"):
    """Derives default streams and threads per stream based on performance hint."""
    cores = max(1, int(cpu_cores))
    batch = max(1, int(batch_size))
    hint_clean = str(hint).lower().strip()

    if hint_clean == "latency":
        streams = 1
    elif hint_clean == "throughput":
        streams = max(1, min(cores, batch))
    else:
        raise ValueError(f"Unknown performance hint: {hint}")

    threads_per_stream = max(1, cores // streams)
    return {
        "num_streams": streams,
        "threads_per_stream": threads_per_stream,
        "total_cores_allocated": streams * threads_per_stream,
    }


def evaluate_batched_throughput(cpu_cores, batch_size, hint="throughput"):
    """Evaluates batched throughput given core topology and performance hints."""
    spec = derive_default_num_streams(cpu_cores, batch_size, hint=hint)
    streams = spec["num_streams"]
    threads = spec["threads_per_stream"]

    thread_efficiency = 1.0 / (1.0 + 0.15 * np.log2(threads))
    stream_concurrency_overhead = 1.0 + 0.05 * (streams - 1)

    work_per_item = 100.0
    total_work = batch_size * work_per_item

    effective_compute_power = (streams * threads * thread_efficiency) / stream_concurrency_overhead
    execution_time = total_work / max(0.001, effective_compute_power)
    items_per_second = batch_size / execution_time

    return {
        "items_per_second": float(items_per_second),
        "execution_time": float(execution_time),
        "streams": streams,
        "threads_per_stream": threads,
    }
