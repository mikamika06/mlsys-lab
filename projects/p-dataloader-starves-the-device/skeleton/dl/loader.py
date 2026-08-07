def compute_min_workers(batch_time, item_time, overhead):
    raise NotImplementedError

def configure_pinning(use_pin_memory, non_blocking):
    raise NotImplementedError

def optimize_hotpath(transform_fn, batch):
    raise NotImplementedError

def evaluate_utilization(loader_time, compute_time):
    raise NotImplementedError
