import sys
sys.path.insert(0, ".")
from inductorsched.fusion import greedy_pointwise_fuse
from inductorsched.memory import compute_memory_usage


def test_no_fusion_across_reduction_boundary():
    nodes = [
        {"id": "buf0", "op": "add", "inputs": [], "shape": (128, 128), "is_pointwise": True},
        {"id": "buf1", "op": "sum", "inputs": ["buf0"], "shape": (128,), "is_pointwise": False},
        {"id": "buf2", "op": "relu", "inputs": ["buf1"], "shape": (128,), "is_pointwise": True},
    ]
    fused = greedy_pointwise_fuse(nodes)
    for group in fused:
        if "buf0" in group:
            assert "buf1" not in group, "Reduction op fused into pointwise group"
            assert "buf2" not in group, "Pointwise fused across reduction boundary"


def test_buffer_reuse_respects_lifetimes():
    nodes = [
        {"id": "buf0", "op": "add", "inputs": [], "shape": (1024,), "is_pointwise": True},
        {"id": "buf1", "op": "relu", "inputs": ["buf0"], "shape": (1024,), "is_pointwise": True},
        {"id": "buf2", "op": "mul", "inputs": ["buf0", "buf1"], "shape": (1024,), "is_pointwise": True},
    ]
    fused = [["buf0"], ["buf1"], ["buf2"]]
    mem_inplace = compute_memory_usage(nodes, fused, inplace_buffers=True)
    mem_no_inplace = compute_memory_usage(nodes, fused, inplace_buffers=False)
    assert mem_inplace <= mem_no_inplace
