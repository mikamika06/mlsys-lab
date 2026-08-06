import torch
from softmaxln.softmax import fused_softmax
from softmaxln.layernorm import fused_layernorm


def test_softmax_sums_to_one():
    x = torch.randn(8, 64)
    out = fused_softmax(x)
    sums = torch.sum(out, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_layernorm_properties():
    x = torch.randn(4, 128)
    weight = torch.ones(128)
    bias = torch.zeros(128)
    out = fused_layernorm(x, (128,), weight=weight, bias=bias)
    means = torch.mean(out, dim=-1)
    vars = torch.var(out, dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4, rtol=1e-4)
    assert torch.allclose(vars, torch.ones_like(vars), atol=1e-2, rtol=1e-2)
