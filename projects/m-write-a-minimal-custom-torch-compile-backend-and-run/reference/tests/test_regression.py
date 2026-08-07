import sys
import torch
sys.path.insert(0, ".")
from backends.attention import compare_attention

def test_attention_numerics():
    model = torch.nn.Linear(16, 16)
    x = torch.randn(2, 16)
    err = compare_attention(model, x)
    assert err < 1e-4, f"error {err} exceeds threshold"
