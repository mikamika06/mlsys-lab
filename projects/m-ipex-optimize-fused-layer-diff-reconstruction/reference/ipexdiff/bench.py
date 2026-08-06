import time


def evaluate_latency_and_speedup(model_fn, input_tensor, num_runs=20):
    """
    Simulates latency measurement for IPEX optimized bf16 vs native PyTorch CPU AMP autocast.
    Calculates execution cycles/latency and speedup ratio.
    """
    # Baseline run
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model_fn(input_tensor, mode="autocast")
    t1 = time.perf_counter()
    autocast_latency = (t1 - t0) / num_runs

    # IPEX run
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model_fn(input_tensor, mode="ipex_bf16")
    t1 = time.perf_counter()
    ipex_latency = (t1 - t0) / num_runs

    speedup = autocast_latency / ipex_latency if ipex_latency > 0 else 1.0
    return {
        "autocast_latency_ms": autocast_latency * 1000.0,
        "ipex_latency_ms": ipex_latency * 1000.0,
        "speedup": speedup,
    }
