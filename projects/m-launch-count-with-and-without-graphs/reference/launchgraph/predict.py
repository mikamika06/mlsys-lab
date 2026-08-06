def predict_speedup(num_ops, kernel_gpu_time_us, host_launch_overhead_us=5.0, graph_launch_overhead_us=2.0):
    """Predict CUDA Graph speedup based on launch overheads and GPU kernel execution times."""
    if num_ops <= 0:
        return 1.0

    standard_time = num_ops * (kernel_gpu_time_us + host_launch_overhead_us)
    graph_time = (num_ops * kernel_gpu_time_us) + graph_launch_overhead_us

    if graph_time <= 0:
        return 1.0

    return standard_time / graph_time
