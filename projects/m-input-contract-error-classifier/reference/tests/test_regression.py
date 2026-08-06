import sys

sys.path.insert(0, ".")
from flash_contract.classifier import classify_inputs

def test_catches_non_contiguous_inner_dim():
    q = {"shape": (2, 128, 8, 64), "strides": (65536, 512, 64, 2), "dtype": "float16"}
    k = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}
    v = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}
    errors = classify_inputs(q, k, v)
    assert "ALIGNMENT_ERROR" in errors

def test_valid_inputs_pass():
    q = {"shape": (2, 128, 8, 64), "strides": (65536, 512, 64, 1), "dtype": "float16"}
    k = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}
    v = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}
    assert classify_inputs(q, k, v) == []

def test_invalid_head_dim():
    q = {"shape": (1, 10, 1, 65), "strides": (650, 65, 65, 1), "dtype": "bfloat16"}
    k = {"shape": (1, 10, 1, 65), "strides": (650, 65, 65, 1), "dtype": "bfloat16"}
    v = {"shape": (1, 10, 1, 65), "strides": (650, 65, 65, 1), "dtype": "bfloat16"}
    errors = classify_inputs(q, k, v)
    assert "HEAD_DIM_ERROR" in errors
