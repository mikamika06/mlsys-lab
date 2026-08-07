import torch
from ckptutils.exactness import verify_gradient_exactness
from ckptutils.pareto import compute_pareto_curve
from ckptutils.breakdown import analyze_op_breakdown


def test_exactness_threshold():
    torch.manual_seed(42)
    model = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    inputs = torch.randn(2, 8)
    strategy = [True]
    err = verify_gradient_exactness(model, inputs, strategy)
    assert err < 1e-4


def test_pareto_output_format():
    torch.manual_seed(42)
    model = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    inputs = torch.randn(1, 4)
    res = compute_pareto_curve(model, inputs, [[False]])
    assert isinstance(res, list)
    assert len(res) == 1


def test_breakdown_structure():
    torch.manual_seed(42)
    model = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    inputs = torch.randn(1, 4)
    b = analyze_op_breakdown(model, inputs, [False])
    assert "total_layers" in b
    assert b["total_layers"] == 1
