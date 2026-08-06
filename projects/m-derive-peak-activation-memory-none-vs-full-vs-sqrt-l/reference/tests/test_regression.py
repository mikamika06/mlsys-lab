import pytest
from ckpt.analysis import peak_activation_memory, optimal_segment_size, recompute_flops_overhead


def test_peak_memory_basic():
    layers = 36
    base_mem = 100.0
    assert peak_activation_memory(layers, base_mem, "full") == 100.0
    assert peak_activation_memory(layers, base_mem, "none") == 3600.0


def test_optimal_segment_size_bounds():
    layers = 64
    base_mem = 50.0
    s = optimal_segment_size(layers, base_mem)
    assert s == 8
    mem = peak_activation_memory(layers, base_mem, "sqrt", segment_size=s)
    assert mem > 0


def test_recompute_overhead_calc():
    layers = 16
    overhead = recompute_flops_overhead(layers, 4)
    assert overhead >= 0.0
