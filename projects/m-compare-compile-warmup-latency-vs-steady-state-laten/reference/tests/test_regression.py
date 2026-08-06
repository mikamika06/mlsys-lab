import torch
import torch.nn as nn
import mpscompile.dynamic

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

def test_dynamic_shape_overhead():
    model = SimpleModel()
    inputs = [torch.randn(4, 32), torch.randn(8, 32)]
    res = mpscompile.dynamic.benchmark_dynamic_shapes(model, inputs)
    assert isinstance(res, dict)
    assert "latencies" in res
    assert len(res["latencies"]) == len(inputs)
    assert res["max_latency"] > 0
