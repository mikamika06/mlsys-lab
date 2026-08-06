import sys

sys.path.insert(0, ".")
from ane_diag.classifier import is_ane_eligible


def test_ane_eligibility_supported_conv2d():
    op = {
        "op_type": "conv2d",
        "dtype": "float16",
        "shape": [1, 64, 224, 224],
        "kernel_height": 3,
        "kernel_width": 3,
    }
    assert is_ane_eligible(op) is True


def test_ane_eligibility_rejects_unsupported_dtype():
    op = {
        "op_type": "conv2d",
        "dtype": "float32",
        "shape": [1, 64, 224, 224],
    }
    assert is_ane_eligible(op) is False


def test_ane_eligibility_rejects_non_batch_one():
    op = {
        "op_type": "relu",
        "dtype": "float16",
        "shape": [2, 64, 224, 224],
    }
    assert is_ane_eligible(op) is False


def test_ane_eligibility_rejects_large_kernel():
    op = {
        "op_type": "conv2d",
        "dtype": "float16",
        "shape": [1, 64, 224, 224],
        "kernel_height": 31,
        "kernel_width": 31,
    }
    assert is_ane_eligible(op) is False


def test_ane_eligibility_rejects_unaligned_matmul():
    op = {
        "op_type": "matmul",
        "dtype": "float16",
        "shape": [1, 7, 64, 64],
    }
    assert is_ane_eligible(op) is False
