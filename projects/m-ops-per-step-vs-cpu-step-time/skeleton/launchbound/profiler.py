"""Module for profiling step times and CPU launch overheads."""

def analyze_step(op_counts, cpu_launch_overhead_us, gpu_time_per_op_us):
    """Analyze total CPU step time, total GPU busy time, and GPU busy fraction."""
    raise NotImplementedError


def predict_small_batch_speedup(op_count, baseline_batch_size, target_batch_size, cpu_launch_overhead_us, gpu_time_per_op_per_batch_us):
    """Predict execution speedup when reducing batch size considering CPU launch bounds."""
    raise NotImplementedError
