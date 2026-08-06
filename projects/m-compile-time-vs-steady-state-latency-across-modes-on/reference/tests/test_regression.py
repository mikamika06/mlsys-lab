import torch
import torch.nn as nn
from compengine.freeze import freeze_and_fold

def test_constant_folding_invariant():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.register_buffer("bias_term", torch.ones(4))

        def forward(self, x):
            return self.linear(x) + self.bias_term

    model = DummyModel()
    processed = freeze_and_fold(model)
    assert processed is not None
    x = torch.randn(2, 4)
    out1 = model(x)
    out2 = processed(x)
    assert torch.allclose(out1, out2, atol=1e-5)
