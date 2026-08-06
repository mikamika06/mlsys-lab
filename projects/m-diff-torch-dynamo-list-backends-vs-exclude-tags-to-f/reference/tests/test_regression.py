import torch
import torch.nn as nn
from dyncomp.metrics import measure_ratios


def test_measure_ratios_positive():
    model = nn.Sequential(nn.Linear(8, 8))
    inputs = [torch.randn(2, 8)]
    res = measure_ratios(model, inputs)
    assert isinstance(res, dict)
    assert "compile_ratio" in res
    assert "run_ratio" in res
    assert res["compile_ratio"] > 0
    assert res["run_ratio"] > 0
