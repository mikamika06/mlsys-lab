import sys
sys.path.insert(0, ".")
from graphbreak.constructs import classify_constructs
from graphbreak.cond import safe_conditional
from graphbreak.partition import build_partition_map
import torch

def test_constructs_classification():
    items = ["tensor_item", "pure_arithmetic"]
    res = classify_constructs(items)
    assert res == ["break", "no_break"]

def test_safe_conditional_output():
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)
    pred = torch.tensor(True)
    res = safe_conditional(pred, x, y)
    assert isinstance(res, torch.Tensor)

def test_partition_map_splitting():
    log = ["step1", "GRAPH BREAK", "step2"]
    res = build_partition_map(log)
    assert len(res) == 2
    assert res[0] == ["step1"]
    assert res[1] == ["step2"]
