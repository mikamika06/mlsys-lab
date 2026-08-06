import os
import shutil
import torch

def run_and_save_cache(compile_dir, artifact_dir):
    """Run compilation and save cache artifacts."""
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_dir
    if os.path.exists(compile_dir):
        shutil.rmtree(compile_dir)
    os.makedirs(compile_dir, exist_ok=True)
    def f(x):
        return x + 1.0
    x = torch.randn(4, 4)
    compiled = torch.compile(f, backend="inductor")
    compiled(x)
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir)
    shutil.copytree(compile_dir, artifact_dir)
    return len(os.listdir(artifact_dir))

def verify_zero_recompiles(artifact_dir, fresh_cache_dir):
    """Verify loading artifacts results in zero recompiles."""
    if os.path.exists(fresh_cache_dir):
        shutil.rmtree(fresh_cache_dir)
    shutil.copytree(artifact_dir, fresh_cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = fresh_cache_dir
    def f(x):
        return x + 1.0
    x = torch.randn(4, 4)
    compiled = torch.compile(f, backend="inductor")
    before = set(os.listdir(fresh_cache_dir))
    compiled(x)
    after = set(os.listdir(fresh_cache_dir))
    return len(after - before) == 0
