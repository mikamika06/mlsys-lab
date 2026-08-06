import sys
sys.path.insert(0, ".")
from inductor_utils.analyzer import count_mlp_kernels

def test_count_mlp_kernels_returns_positive_integer():
    configs = [{"fused": True, "layers": 2}, {"fused": False, "layers": 2}]
    res = count_mlp_kernels(configs)
    assert isinstance(res, int)
    assert res > 0

def test_count_mlp_kernels_deterministic():
    configs = [{"fused": True, "layers": 3}]
    assert count_mlp_kernels(configs) == count_mlp_kernels(configs)
