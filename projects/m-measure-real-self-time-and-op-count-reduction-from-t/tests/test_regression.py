import sys
sys.path.insert(0, ".")
from torchprof.fusion import count_fused_kernels


def test_fusion_ratio_bound():
    nodes = [{"name": "fused_add_mul", "is_fused": True}, {"name": "sin_kernel", "is_fused": False}]
    res = count_fused_kernels(nodes)
    assert res["fused"] >= 1
    assert res["unfused"] >= 1
