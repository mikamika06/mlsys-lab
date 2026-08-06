"""GPU utilization calculator."""

def compute_profile_utilization(kernels, trace_start_ns, trace_end_ns):
    """Compute active GPU utilization ratio merging overlapping kernel intervals."""
    raise NotImplementedError


def compare_gpu_utilization(report_a, report_b):
    """Compare GPU utilization between two reports and return argmin index."""
    raise NotImplementedError
