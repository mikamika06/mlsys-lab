import time
import torch

def measure_speedup(model, inputs):
    model_eager = model.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(10):
            _ = model_eager(*inputs)
        t1 = time.time()
    eager_time = max(t1 - t0, 1e-6)

    try:
        compiled = torch.compile(model_eager, backend="aot_eager")
        with torch.no_grad():
            t2 = time.time()
            for _ in range(10):
                _ = compiled(*inputs)
            t3 = time.time()
        comp_time = max(t3 - t2, 1e-6)
    except Exception:
        comp_time = eager_time

    return eager_time / comp_time
