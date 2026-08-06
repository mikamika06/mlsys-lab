import time
import torch

def measure_warmup(model, x):
    """Measures warmup vs steady state latency."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = model.to(device)
    inp = x.to(device)

    try:
        compiled = torch.compile(m)
    except Exception:
        compiled = m

    start = time.perf_counter()
    _ = compiled(inp)
    if device.type == "mps":
        torch.mps.synchronize()
    warmup = time.perf_counter() - start

    start = time.perf_counter()
    _ = compiled(inp)
    if device.type == "mps":
        torch.mps.synchronize()
    steady = time.perf_counter() - start

    if steady == 0:
        steady = 1e-6
    return {"warmup_latency": warmup, "steady_latency": steady, "ratio": warmup / steady}
