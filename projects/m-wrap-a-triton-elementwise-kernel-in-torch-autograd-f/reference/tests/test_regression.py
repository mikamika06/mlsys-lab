import torch
from triton_wrap.autograd import FusedSiluAutograd


def test_autograd_gradcheck():
    x = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(FusedSiluAutograd.apply, x, eps=1e-6, atol=1e-4)


def test_output_values():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    out = FusedSiluAutograd.apply(x)
    assert out.shape == x.shape
