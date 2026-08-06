import sys
import torch
sys.path.insert(0, ".")
from cpuharness.modes import compare_cpu_modes
from cpuharness.safety import verify_fullgraph_capture

def test_modes_return_dict():
    mod = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
    inputs = (torch.randn(4, 16),)
    res = compare_cpu_modes(mod, inputs)
    assert isinstance(res, dict)
    assert "eager" in res
    assert "default" in res

def test_fullgraph_unsupported_exception():
    def bad_model(x):
        if x.sum() > 0:
            return x * 2
        return x * 3
    inputs = (torch.randn(4, 16),)
    exc_type = verify_fullgraph_capture(bad_model, inputs)
    assert exc_type is not None
