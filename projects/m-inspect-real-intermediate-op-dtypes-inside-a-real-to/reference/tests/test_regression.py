import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autocast_inspect.inspector import synthesize_overflow, inspect_autocast
import torch

def test_overflow_actually_overflows_fp16():
    a, b = synthesize_overflow()
    val = (a.half() * b.half()).sum().item()
    assert val == float('inf') or val == float('-inf'), "Did not overflow in fp16"

def test_overflow_does_not_overflow_bf16():
    a, b = synthesize_overflow()
    val = (a.bfloat16() * b.bfloat16()).sum().item()
    assert val != float('inf') and val != float('-inf'), "Overflowed in bf16 too"

def test_inspect_returns_correct_shape():
    model = torch.nn.Linear(10, 10)
    x = torch.randn(2, 10)
    res = inspect_autocast(model, x, "cpu", torch.bfloat16)
    assert "output_dtype" in res
    assert "activation_dtypes" in res
    assert "weight_dtypes" in res
    assert res["weight_dtypes"][0] == torch.float32
