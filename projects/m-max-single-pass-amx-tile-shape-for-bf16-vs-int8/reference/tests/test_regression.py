import sys

sys.path.insert(0, ".")
from amxtile.shape import max_tile_shape
from amxtile.throughput import tmul_vs_avx512_ratio
from amxtile.classify import classify_tileability


def test_tile_shape_dimensions():
    for dt in ["bf16", "int8"]:
        rows, cols = max_tile_shape(dt)
        assert rows == 16, f"rows must be 16, got {rows}"
        assert cols in (32, 64), f"invalid cols {cols}"


def test_throughput_ratio_positive():
    for dt in ["bf16", "int8"]:
        ratio = tmul_vs_avx512_ratio(dt)
        assert ratio > 1.0, f"ratio must exceed 1.0, got {ratio}"


def test_classification_logic():
    res = classify_tileability(16, 32, 64, "bf16")
    assert res["single_pass"] is True
    res_large = classify_tileability(64, 128, 64, "bf16")
    assert res_large["single_pass"] is False
