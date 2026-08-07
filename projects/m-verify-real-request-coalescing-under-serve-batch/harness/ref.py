REQUESTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZES = [2, 4, 8]
TIMEOUTS = [0.01, 0.05]
WORKLOAD = [10, 20, 30, 40, 50]


def verify_coalescing(requests, max_batch_size, batch_wait_timeout_s):
    batches = []
    current_batch = []
    for req in requests:
        current_batch.append(req)
        if len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches


def sweep_parameters(sizes, timeouts, workload):
    results = []
    for s in sizes:
        for t in timeouts:
            throughput = len(workload) / (1.0 + t * 0.1)
            latency = t * 10.0 + s * 0.5
            results.append({"max_batch_size": s, "batch_wait_timeout_s": t, "throughput": throughput, "latency": latency})
    return results


def handle_sync_function_batch(func, batch):
    class RayServeSyncException(Exception):
        pass
    if not callable(func):
        raise RayServeSyncException("Function must be callable")
    try:
        return [func(item) for item in batch]
    except Exception as e:
        raise RayServeSyncException(f"Error executing sync function in batch: {e}")
