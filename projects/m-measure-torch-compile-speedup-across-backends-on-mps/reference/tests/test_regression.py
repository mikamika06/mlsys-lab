import sys
import torch
sys.path.insert(0, ".")
from compilebench.verify import check_equivalence
from compilebench.breakfinder import identify_graph_break
from compilebench.measure import measure_speedup

def test_equivalence_passes():
    x = torch.randn(2, 16, 64)
    assert check_equivalence(x, x, 1e-5)

def test_breakfinder_detects():
    logs = ["Graph break due to unsupported tensor shape at line 42"]
    res = identify_graph_break(logs)
    assert "line 42" in res or "unsupported" in res

def test_measure_returns_float():
    mod = torch.nn.Linear(64, 64)
    x = torch.randn(2, 64)
    rate = measure_speedup(mod, x, warmup=1, steps=2)
    assert isinstance(rate, float)
    assert rate > 0.0
