import ref
import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = False
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        if not self.compiled:
            import time
            time.sleep(0.01)
            self.compiled = True
        return self.linear(x)

def check(workdir):
    from bench.core import benchmark_compiled_step
    model = DummyModel()
    inputs = (torch.randn(4, 16),)
    times = benchmark_compiled_step(model, inputs, warmup=2, iters=3)
    out = {"compile_excluded": 0.0}
    if isinstance(times, list) and len(times) == 3:
        if all(t < 0.005 for t in times):
            out["compile_excluded"] = 1.0
        else:
            out["_note"] = f"latencies include compilation overhead: {times}"
    else:
        out["_note"] = f"invalid return format: {times}"
    return out
