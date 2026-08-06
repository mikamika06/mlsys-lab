import time


def measure_eager_latency(model, inputs, warmup, runs):
    """Measure eager PyTorch execution time in milliseconds."""
    for _ in range(warmup):
        model(*inputs)
    start = time.perf_counter()
    for _ in range(runs):
        model(*inputs)
    end = time.perf_counter()
    return ((end - start) / runs) * 1000.0


def measure_tvm_latency(compiled_module, inputs, warmup, runs):
    """Measure TVM compiled execution time in milliseconds."""
    for _ in range(warmup):
        compiled_module(*inputs)
    start = time.perf_counter()
    for _ in range(runs):
        compiled_module(*inputs)
    end = time.perf_counter()
    return ((end - start) / runs) * 1000.0


def compute_latency_ratio(model, compiled_module, inputs, warmup=5, runs=20):
    """Compute the ratio of eager latency over TVM latency."""
    eager_ms = measure_eager_latency(model, inputs, warmup, runs)
    tvm_ms = measure_tvm_latency(compiled_module, inputs, warmup, runs)
    if tvm_ms <= 0:
        return 0.0
    return eager_ms / tvm_ms
