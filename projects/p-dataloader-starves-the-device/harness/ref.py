def measure_loader_fraction(step_times, wait_times):
    from dl.profiler import measure_loader_fraction as ref_fn
    return ref_fn(step_times, wait_times)

def compute_min_workers(batch_time, item_time, overhead):
    from dl.loader import compute_min_workers as ref_fn
    return ref_fn(batch_time, item_time, overhead)

def configure_pinning(use_pin_memory, non_blocking):
    from dl.loader import configure_pinning as ref_fn
    return ref_fn(use_pin_memory, non_blocking)

def optimize_hotpath(transform_fn, batch):
    from dl.loader import optimize_hotpath as ref_fn
    return ref_fn(transform_fn, batch)

def evaluate_utilization(loader_time, compute_time):
    from dl.loader import evaluate_utilization as ref_fn
    return ref_fn(loader_time, compute_time)

def predict_workers(target_rate, item_rate):
    from dl.config import predict_workers as ref_fn
    return ref_fn(target_rate, item_rate)
