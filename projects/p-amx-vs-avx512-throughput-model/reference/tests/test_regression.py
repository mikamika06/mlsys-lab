import sys
sys.path.insert(0, ".")
from amx_model.model import select_best_isa, predict_amx, predict_avx512


def test_selection_non_trivial():
    assert select_best_isa(256, 256, 256, "int8") == "amx"


def test_throughput_positive():
    assert predict_amx(64, 64, 64, "int8") > 0.0
    assert predict_avx512(64, 64, 64, "int8") > 0.0
