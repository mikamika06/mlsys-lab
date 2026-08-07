def compute_min_workers(batch_time, item_time, overhead):
    import math
    needed = batch_time / max(1e-6, item_time)
    return max(1, math.ceil(needed - overhead))

def configure_pinning(use_pin_memory, non_blocking):
    return bool(use_pin_memory and non_blocking)

def optimize_hotpath(transform_fn, batch):
    return transform_fn(batch)

def evaluate_utilization(loader_time, compute_time):
    max_t = max(loader_time, compute_time)
    if max_t == 0:
        return 1.0
    return compute_time / max_t
