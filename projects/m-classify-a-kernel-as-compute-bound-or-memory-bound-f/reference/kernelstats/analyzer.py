from kernelstats.metrics import compute_arithmetic_intensity


def classify_kernel(intensity, ridge_point):
    if intensity >= ridge_point:
        return "compute-bound"
    return "memory-bound"


def analyze_trace(trace_record, hardware_spec):
    flops = trace_record.get("flops", 0)
    bytes_tx = trace_record.get("bytes_transferred", 0)
    ridge = hardware_spec.get("ridge_point", 0.0)
    intensity = compute_arithmetic_intensity(flops, bytes_tx)
    classification = classify_kernel(intensity, ridge)
    return {
        "kernel_name": trace_record.get("name", "unknown"),
        "arithmetic_intensity": intensity,
        "classification": classification,
    }
