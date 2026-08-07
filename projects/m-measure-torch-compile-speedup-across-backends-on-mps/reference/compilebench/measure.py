import time
import torch

def measure_speedup(model, inputs, backend="eager", warmup=2, steps=5):
    device = inputs.device
    for _ in range(warmup):
        _ = model(inputs)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        _ = model(inputs)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return steps / max(elapsed, 1e-6)
