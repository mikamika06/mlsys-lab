import sys
import torch

sys.path.insert(0, ".")
from autoperf.nested import run_with_nested_disable


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


def test_nested_disable_forces_false():
    model = DummyModel()
    x = torch.randn(4, 16)
    states, _ = run_with_nested_disable(model, x)
    assert states[0] is True
    assert states[1] is False
    assert states[2] is True


def test_nested_compute_dtype_is_fp32():
    model = DummyModel()
    x = torch.randn(4, 16)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        with torch.autocast(device_type=device_type, enabled=False):
            assert torch.get_autocast_dtype(device_type) == torch.float32 or not torch.is_autocast_enabled(device_type)
