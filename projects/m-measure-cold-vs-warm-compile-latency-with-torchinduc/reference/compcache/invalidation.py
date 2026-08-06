import os
import shutil
import torch

def check_cache_behavior(cache_dir):
    """Check cache behavior on rerun and constant change."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    val = 1.0
    def model(x):
        return x * val
    x = torch.randn(4, 4)
    c1 = torch.compile(model, backend="inductor")
    c1(x)
    n1 = len(os.listdir(cache_dir))
    val = 1.0
    c2 = torch.compile(model, backend="inductor")
    c2(x)
    n2 = len(os.listdir(cache_dir))
    val = 2.0
    c3 = torch.compile(model, backend="inductor")
    c3(x)
    n3 = len(os.listdir(cache_dir))
    return {"identical_hit": n2 == n1, "invalidated": n3 > n2}
