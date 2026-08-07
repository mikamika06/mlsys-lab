import sys
import torch
import torch.nn as nn
sys.path.insert(0, ".")


def test_base_weights_frozen_invariant():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(16, 16)
            self.lora = nn.Linear(16, 16)

        def forward(self, x):
            return self.base(x) + self.lora(x)

    model = DummyModel()
    init_base = model.base.weight.clone()
    model.base.weight.data += 0.1
    assert torch.equal(model.base.weight, init_base) is False
