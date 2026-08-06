WORKLOADS = [
    {
        "tensors": [([1, 4096], "fp16"), ([4096, 4096], "int4"), ([1, 4096], "fp16")],
        "flops": 2 * 1 * 4096 * 4096,
        "peak_bandwidth_gbps": 2000.0,
        "peak_tflops": 312.0,
        "measured_time_sec": 0.000018,
    },
    {
        "tensors": [([32, 2048], "fp16"), ([2048, 8192], "fp16"), ([32, 8192], "fp16")],
        "flops": 2 * 32 * 2048 * 8192,
        "peak_bandwidth_gbps": 1500.0,
        "peak_tflops": 120.0,
        "measured_time_sec": 0.000075,
    },
    {
        "tensors": [([512, 512], "fp32"), ([512, 512], "fp32"), ([512, 512], "fp32")],
        "flops": 2 * 512 * 512 * 512,
        "peak_bandwidth_gbps": 900.0,
        "peak_tflops": 40.0,
        "measured_time_sec": 0.000015,
    },
    {
        "tensors": [([1, 16384], "int8"), ([16384], "int8")],
        "flops": 16384,
        "peak_bandwidth_gbps": 3200.0,
        "peak_tflops": 600.0,
        "measured_time_sec": 0.00000002,
    },
]


def reference_compute_analytic(tensors, flops, peak_bandwidth_gbps, peak_tflops):
    dtype_map = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
    total_bytes = sum(
        dtype_map[dtype] * float(int(np_prod(shape)))
        for shape, dtype in tensors
    )
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


def reference_analyze_measured(measured_time_sec, total_bytes, flops, peak_bandwidth_gbps, peak_tflops):
    time_mem = total_bytes / (peak_bandwidth_gbps * 1e9)
    time_compute = flops / (peak_tflops * 1e12)
    analytic_time = max(time_mem, time_compute)

    achieved_gbps = (total_bytes / measured_time_sec) / 1e9
    achieved_tflops = (flops / measured_time_sec) / 1e12
    bandwidth_utilization = achieved_gbps / peak_bandwidth_gbps
    compute_utilization = achieved_tflops / peak_tflops
    rel_err = abs(measured_time_sec - analytic_time) / analytic_time

    return {
        "achieved_gbps": achieved_gbps,
        "achieved_tflops": achieved_tflops,
        "bandwidth_utilization": bandwidth_utilization,
        "compute_utilization": compute_utilization,
        "rel_err": rel_err,
        "efficiency_ratio": analytic_time / measured_time_sec,
    }


def np_prod(shape):
    res = 1
    for s in shape:
        res *= s
    return res
