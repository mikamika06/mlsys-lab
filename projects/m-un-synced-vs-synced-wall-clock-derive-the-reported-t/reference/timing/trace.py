def find_missing_cuda_synchronize(trace_events):
    """Find the index of the event where torch.cuda.synchronize is missing."""
    for idx, event in enumerate(trace_events):
        if event.get("type") == "gpu_idle" and event.get("duration", 0) > event.get("threshold", 100):
            if not event.get("has_sync", False):
                return idx
    return -1
