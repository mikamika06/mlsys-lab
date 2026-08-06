import sys

sys.path.insert(0, ".")
from ggml_isa.parser import parse_isa_flags
from ggml_isa.benchmark import compare_performance
from ggml_isa.classifier import classify_format


def test_parse_valid_flags():
    log = "cmake -DGGML_AVX512=ON -DGGML_AMX=OFF .."
    flags = parse_isa_flags(log)
    assert flags.get("GGML_AVX512") is True
    assert flags.get("GGML_AMX") is False


def test_performance_comparison():
    res = compare_performance(150.0, 100.0)
    assert res["speedup"] == 1.5
    assert res["efficient"] is True


def test_classifier_distinctions():
    assert classify_format("Q4_0") == "standard"
    assert classify_format("Q8_0") == "accelerated"
    assert classify_format("F32") == "fallback"
