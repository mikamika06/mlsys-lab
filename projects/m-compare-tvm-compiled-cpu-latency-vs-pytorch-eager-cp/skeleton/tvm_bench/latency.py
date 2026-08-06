def measure_eager_latency(model, inputs, warmup, runs):
    """Measure eager PyTorch execution time in milliseconds."""
    raise NotImplementedError


def measure_tvm_latency(compiled_module, inputs, warmup, runs):
    """Measure TVM compiled execution time in milliseconds."""
    raise NotImplementedError


def compute_latency_ratio(model, compiled_module, inputs, warmup=5, runs=20):
    """Compute the ratio of eager latency over TVM latency."""
    raise NotImplementedError
