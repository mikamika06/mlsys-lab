import sys
sys.path.insert(0, ".")
from quantizer.oneshot import run_oneshot
from quantizer.onloading import evaluate_onloading_impact
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)


def test_oneshot_returns_valid_dict():
    model = DummyModel()
    res = run_oneshot(model, sequential_onloading=False)
    assert isinstance(res, dict)
    assert len(res) > 0
    for k, v in res.items():
        assert "quantized" in v
        assert "scale" in v


def test_onloading_impact_structure():
    model = DummyModel()
    impact = evaluate_onloading_impact(model)
    assert isinstance(impact, dict)
    assert "peak_memory_on" in impact
    assert "peak_memory_off" in impact
    assert impact["peak_memory_on"] < impact["peak_memory_off"]
