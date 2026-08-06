import time
import torch


def measure_latency_cliff(op_fallback, op_native, sample_input):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = sample_input.to(device)
    start = time.perf_counter()
    for _ in range(20):
        _ = op_fallback(x)
    t_fallback = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20):
        _ = op_native(x)
    t_native = time.perf_counter() - start
    ratio = t_fallback / max(t_native, 1e-6)
    return {"fallback_time": t_fallback, "native_time": t_native, "ratio": ratio}
