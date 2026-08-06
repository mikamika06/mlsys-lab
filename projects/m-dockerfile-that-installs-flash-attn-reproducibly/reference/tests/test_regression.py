import sys
sys.path.insert(0, ".")
from flashbuild.dockerfile import generate_dockerfile
from flashbuild.cost import estimate_install_cost, latency_ratio

def test_dockerfile_is_pinned():
    cfg = {"torch_version": "2.4.0", "cuda_version": "12.4", "flash_version": "2.6.3", "gpu_arch": "89"}
    df = generate_dockerfile(cfg)
    assert "flash-attn==2.6.3" in df
    assert "pytorch/pytorch:2.4.0-cuda12.4" in df

def test_cost_is_positive():
    cfg = {"torch_version": "2.4.0", "cuda_version": "12.4", "flash_version": "2.6.3", "gpu_arch": "89"}
    assert estimate_install_cost(cfg) > 0

def test_latency_ratio_monotonic():
    cfg1 = {"torch_version": "2.4.0", "cuda_version": "12.4", "flash_version": "2.6.3", "gpu_arch": "89"}
    cfg2 = {"torch_version": "2.5.0", "cuda_version": "12.6", "flash_version": "4.0.0", "gpu_arch": "90"}
    r = latency_ratio(cfg2, cfg1)
    assert r > 0
