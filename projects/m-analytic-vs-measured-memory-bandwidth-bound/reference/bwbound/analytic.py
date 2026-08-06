DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def compute_analytic_bound(tensors, flops, peak_bandwidth_gbps, peak_tflops):
    """Compute analytic roofline execution time bound and bottleneck breakdown."""
    total_bytes = 0.0
    for shape, dtype in tensors:
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        bytes_per_elem = DTYPE_BYTES[dtype]
        total_bytes += num_elements * bytes_per_elem

    time_mem = total_bytes / (peak_bandwidth_gbps * 1e9)
    time_compute = flops / (peak_tflops * 1e12)
    analytic_time = max(time_mem, time_compute)
    arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
    is_memory_bound = time_mem >= time_compute

    return {
        "total_bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "time_mem_sec": time_mem,
        "time_compute_sec": time_compute,
        "analytic_time_sec": analytic_time,
        "is_memory_bound": is_memory_bound,
    }
