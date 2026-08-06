import sys
import torch
sys.path.insert(0, ".")
from peftcomp.analyzer import extract_graph_breaks
from peftcomp.warmup import measure_warmup_cost
from peftcomp.recompile import check_recompilation_triggers


class DummyPEFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.adapter = torch.nn.Linear(16, 16)

    def forward(self, x):
        h = self.linear(x)
        if x.shape[1] > 4:
            torch._dynamo.graph_break()
        return h + self.adapter(h)


def test_graph_break_detection():
    model = DummyPEFTModel()
    x = torch.randn(1, 8, 16)
    res = extract_graph_breaks(model, x)
    assert res["count"] >= 1


def test_warmup_gap_positive():
    model = DummyPEFTModel()
    compiled = torch.compile(model)
    x = torch.randn(1, 2, 16)
    res = measure_warmup_cost(compiled, x)
    assert res["warmup_gap_ns"] >= 0


def test_recompilation_behavior():
    model = DummyPEFTModel()
    compiled = torch.compile(model)
    base = torch.randn(1, 2, 16)
    varied = torch.randn(1, 10, 16)
    res = check_recompilation_triggers(compiled, base, varied)
    assert res["varied_recompiled"] is True
