def analyze_roofline(op_mix, spec):
    total_flops = sum(op["flops"] for op in op_mix)
    total_bytes = sum(op["bytes"] for op in op_mix)
    ai = total_flops / max(total_bytes, 1.0)
    knee_point = (spec["peak_tflops"] * 1e12) / (spec["peak_bandwidth_gbps"] * 1e9)
    attained_tflops = min(spec["peak_tflops"], (ai * spec["peak_bandwidth_gbps"] * 1e9) / 1e12)
    bound = "compute_bound" if ai >= knee_point else "memory_bound"
    return {
        "intensity": ai,
        "knee_point": knee_point,
        "attained_tflops": attained_tflops,
        "bound": bound
    }
