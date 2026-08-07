import torch
import sys
sys.path.insert(0, ".")
from triton_matmul.kernel import matmul_grouped


def test_matmul_shapes():
    x = torch.randn(256, 256, device="cuda" if torch.cuda.is_available() else "cpu")
    y = torch.randn(256, 256, device="cuda" if torch.cuda.is_available() else "cpu")
    out = matmul_grouped(x, y)
    assert out.shape == (256, 256)


def test_matmul_values():
    x = torch.ones(128, 128, device="cuda" if torch.cuda.is_available() else "cpu")
    y = torch.ones(128, 128, device="cuda" if torch.cuda.is_available() else "cpu")
    out = matmul_grouped(x, y)
    assert torch.allclose(out, torch.full_like(out, 128.0))
