import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from optbits.measure import measure_optimizer_bytes
from optbits.train import train_short_loop
from optbits.memory import per_parameter_memory


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        return self.linear(x)


def test_optimizer_byte_sizes():
    model = SimpleModel()
    res = measure_optimizer_bytes(model)
    assert "torch_adamw" in res
    assert "adamw_8bit" in res
    assert res["adamw_8bit"] < res["torch_adamw"]
    assert 0.2 <= res["size_ratio"] <= 0.35


def test_training_convergence():
    torch.manual_seed(42)
    model = SimpleModel()
    x = torch.randn(32, 16)
    y = torch.randn(32, 1)
    init_loss, final_loss = train_short_loop(model, x, y, steps=5)
    assert final_loss < init_loss


def test_per_parameter_memory_structure():
    model = SimpleModel()
    mem = per_parameter_memory(model)
    for name, info in mem.items():
        assert info["adam8bit_bytes"] < info["adam32bit_bytes"]
        assert info["numel"] > 0
