import os
import time
import shutil
import torch

def measure_compile_latencies(cache_dir):
    """Measure cold vs warm compile latency."""
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    def model(x):
        return torch.sin(x) + torch.cos(x)
    x = torch.randn(16, 16)
    c1 = torch.compile(model, backend="inductor")
    t0 = time.time()
    _ = c1(x)
    cold_t = time.time() - t0
    c2 = torch.compile(model, backend="inductor")
    t1 = time.time()
    _ = c2(x)
    warm_t = time.time() - t1
    return cold_t, warm_t
