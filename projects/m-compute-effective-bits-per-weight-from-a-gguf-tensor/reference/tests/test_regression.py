import pytest
from gguf_utils.bits import compute_effective_bits
from gguf_utils.perf import fit_decode_performance
from gguf_utils.policy import choose_model


def test_effective_bits_basic():
    tensors = [
        {"n_elements": 1000, "bits_per_weight": 4.0},
        {"n_elements": 1000, "bits_per_weight": 8.0},
    ]
    assert compute_effective_bits(tensors) == 6.0


def test_fit_decode_basic():
    data = [
        {"bpw": 4.0, "tok_s": 40.0},
        {"bpw": 8.0, "tok_s": 20.0},
    ]
    res = fit_decode_performance(data)
    assert "slope" in res
    assert "intercept" in res


def test_choose_model_basic():
    options = [
        {"name": "8B-Q8", "memory_gb": 9.0, "score": 85.0},
        {"name": "14B-Q4", "memory_gb": 8.5, "score": 90.0},
    ]
    assert choose_model(10.0, options) == "14B-Q4"
    assert choose_model(8.7, options) == "14B-Q4"
