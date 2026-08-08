def compute_gpu_utilization(kernel_events, capture_window):
    """Compute active GPU utilization percentage from kernel execution events."""
    raise NotImplementedError


def compare_batch_utilizations(captures):
    """Compare GPU utilization percentages and return argmin/argmax index based on efficiency."""
    raise NotImplementedError
