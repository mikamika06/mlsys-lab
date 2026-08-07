import sys

sys.path.insert(0, ".")
from zero_estimator.memory import calculate_sharded_elements, calculate_prefetch_depth


def test_fsdp_padding_overhead_is_real():
    layers = [1025, 1025, 1025]
    fsdp, zero3 = calculate_sharded_elements(layers, 8)
    assert fsdp > zero3


def test_prefetch_depth_accumulation():
    layers = [1000000000, 1000000000, 1000000000, 1000000000]
    bw = 1000000000.0
    computes = [1.0, 1.0, 1.0, 1.0]
    g_times, depths = calculate_prefetch_depth(layers, computes, bw)
    assert depths[3] == 2
