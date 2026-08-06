import time
import torch

def benchmark_dynamic_shapes(model, inputs):
    """Benchmarks recompilation cost across shape changes."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = model.to(device)

    try:
        compiled = torch.compile(m)
    except Exception:
        compiled = m

    latencies = []
    for inp in inputs:
        t_inp = inp.to(device)
        start = time.perf_counter()
        _ = compiled(t_inp)
        if device.type == "mps":
            torch.mps.synchronize()
        latencies.append(time.perf_counter() - start)

    return {"latencies": latencies, "max_latency": max(latencies), "min_latency": min(latencies)}
