import sys
sys.path.insert(0, ".")
from fusion.analyzer import classify_kernels, count_vectorized_loops, check_fusion_validity
import ref


def test_classify_kernels_not_empty():
    res = classify_kernels(ref.KERNELS_DUMP)
    assert len(res) > 0, "No kernels found in dump"
    for k, v in res.items():
        assert v in ("reduction", "pointwise", "unknown")


def test_vectorized_loops_positive():
    count = count_vectorized_loops(ref.CPP_DUMP)
    assert count > 0, "Expected vectorized loops to be counted"


def test_fusion_validity_flag():
    assert check_fusion_validity({"fused": True}) is True
    assert check_fusion_validity({"fused": False}) is False
