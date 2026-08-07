import time
import torch


def measure_speedup(model: torch.nn.Module, x: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        start = time.time()
        for _ in range(10):
            _ = model(x)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        baseline = time.time() - start
        try:
            compiled = torch.compile(model, backend="eager")
            start = time.time()
            for _ in range(10):
                _ = compiled(x)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            comp_time = time.time() - start
        except Exception:
            comp_time = baseline
        if comp_time <= 0:
            return 1.0
        return float(baseline / comp_time)
