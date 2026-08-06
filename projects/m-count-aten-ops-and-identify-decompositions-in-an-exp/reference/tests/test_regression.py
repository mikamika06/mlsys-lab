import sys
import torch

sys.path.insert(0, ".")
from atenaudit.analyzer import count_aten_ops


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1.0)


def test_count_aten_ops_structure():
    model = SimpleModel()
    x = torch.randn(2, 2)
    ep = torch.export.export(model, (x,))
    counts = count_aten_ops(ep)
    assert isinstance(counts, dict)
    for op, cnt in counts.items():
        assert isinstance(op, str)
        assert isinstance(cnt, int)
        assert cnt > 0
