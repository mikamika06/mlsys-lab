import torch
from softmax.autograd import fused_softmax_autograd

def test_softmax_correctness():
    x = torch.randn(16, 64, requires_grad=True)
    y = fused_softmax_autograd(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_softmax_values():
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
    y = fused_softmax_autograd(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(y, expected, atol=1e-5)
