import time
import torch

def measure_latencies(model, inputs, num_warmup=2, num_iters=5):
    for _ in range(num_warmup):
        _ = model(*inputs)
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(*inputs)
        torch.cpu.synchronize() if hasattr(torch.cpu, "synchronize") else None
    end = time.perf_counter()
    return (end - start) / num_iters
