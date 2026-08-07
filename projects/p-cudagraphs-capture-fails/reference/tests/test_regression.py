import torch
from model.net import evaluate_with_graph

def test_graph_output_shape():
    x = torch.randn(4, 16)
    out = evaluate_with_graph(x)
    assert out.shape == (4, 16)

def test_graph_output_values():
    x = torch.ones(2, 16)
    out = evaluate_with_graph(x)
    assert torch.allclose(out, x)
