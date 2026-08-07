import time

def run_concurrent_batch(requests, adapters):
    active_adapters = set()
    total_tokens = 0
    start_time = time.time()
    for req in requests:
        adapter_id = req.get("adapter_id")
        if adapter_id is not None:
            active_adapters.add(adapter_id)
        total_tokens += req.get("tokens", 1)
    duration = max(time.time() - start_time, 0.001)
    throughput = total_tokens / duration
    return {
        "active_adapters": len(active_adapters),
        "throughput": throughput,
        "total_tokens": total_tokens,
        "duration": duration
    }
