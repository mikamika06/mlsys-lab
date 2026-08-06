import torch
from peftutils.size import compute_adapter_bytes, compute_storage_ratio
from peftutils.verify import save_and_verify_adapter


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x, weights):
        return self.linear(x) + weights.get("lora", torch.zeros_like(x))


def test_adapter_byte_calculation():
    cfg = {"r": 4, "hidden_size": 16, "intermediate_size": 32}
    b = compute_adapter_bytes(cfg, ["q_proj"], dtype_bytes=2)
    assert b == (4 * 16 + 16 * 4) * 2


def test_storage_ratio():
    ratio = compute_storage_ratio(100, 1000)
    assert ratio == 0.1


def test_verify_adapter():
    model = SimpleModel()
    weights = {"lora": torch.ones(1, 16)}
    x = torch.zeros(1, 16)
    assert save_and_verify_adapter(model, weights, x) is True
