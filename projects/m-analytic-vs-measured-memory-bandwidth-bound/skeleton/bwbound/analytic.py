DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def compute_analytic_bound(tensors, flops, peak_bandwidth_gbps, peak_tflops):
    """Compute analytic roofline execution time bound and bottleneck breakdown."""
    raise NotImplementedError
