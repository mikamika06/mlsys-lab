import torch
from gcapture.branch import inspect_baked_branch
from gcapture.histogram import aten_op_histogram


class BranchyModel(torch.nn.Module):

    def forward(self, x):
        if x.mean() > 0.0:
            return x * 2.0
        return x + 5.0


def test_branch_baking_detection():
    mod = BranchyModel()
    pos = torch.tensor([1.0, 2.0])
    neg = torch.tensor([-1.0, -2.0])

    res = inspect_baked_branch(mod, pos, neg)
    assert res["trace_baked_branch"] is True
    assert res["trace_took_branch"] == "then_branch"


def test_aten_histogram_counts():
    class SimpleModule(torch.nn.Module):

        def forward(self, x, y):
            return torch.add(x, y) * y

    mod = SimpleModule()
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    ep = torch.export.export(mod, (x, y))
    hist = aten_op_histogram(ep)
    assert sum(hist.values()) >= 2
