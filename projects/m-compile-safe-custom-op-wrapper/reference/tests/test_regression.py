import torch
from custom_op.wrapper import run_attention

def test_compile_fullgraph_and_fake():
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    compiled = torch.compile(run_attention, fullgraph=True)
    out = compiled(q, k, v)
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape
