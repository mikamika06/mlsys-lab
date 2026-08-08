def estimate_draft_latency(param_count, precision, hw_config):
    bytes_per_elem = 2.0 if precision == "fp16" else 1.0
    weight_bytes = param_count * bytes_per_elem
    mem_time = weight_bytes / (hw_config["mem_bandwidth_gbps"] * 1e9)

    flops = 2.0 * param_count
    peak_flops = hw_config["fp16_flops"] if precision == "fp16" else hw_config["int8_flops"]
    compute_time = flops / peak_flops

    bounded_time = max(mem_time, compute_time)

    overhead = hw_config["base_launch_latency_s"]
    if precision == "int8":
        overhead += hw_config["quant_overhead_per_param_s"] * param_count

    return bounded_time + overhead
