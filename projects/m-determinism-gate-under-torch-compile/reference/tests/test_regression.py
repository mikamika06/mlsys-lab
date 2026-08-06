import sys
import torch

sys.path.insert(0, ".")
from detgate.core import check_determinism, stabilized_gate


def test_determinism_gate_catches_variance():
    class FlakyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)
            self.counter = 0

        def forward(self, x):
            self.counter += 1
            out = self.linear(x)
            if self.counter > 2:
                out = out + 1e-6
            return out

    model = FlakyModule()
    x = torch.randn(2, 8)
    res = check_determinism(model, (x,), num_runs=4)
    assert res is False, "failed to catch non-deterministic outputs"


def test_stabilized_gate_runs_warmup():
    class WarmupModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    model = WarmupModule()
    x = torch.randn(2, 8)
    res = stabilized_gate(model, (x,), warmup_runs=1, test_runs=3)
    assert res is True, "stabilized gate failed on deterministic module"
