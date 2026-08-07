import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from peft_verify.verify import check_modules_to_save_trainable
from peft_verify.size import compute_adapter_size_ratio


class MockBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 32)
        self.head = nn.Linear(32, 10)


def test_check_modules_to_save_trainable():
    model = MockBaseModel()
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    res = check_modules_to_save_trainable(model, ["head"])
    assert res["base_frozen"] is True
    assert res["head_trainable"] is True
    assert res["valid"] is True


def test_detects_unfrozen_base():
    model = MockBaseModel()
    for param in model.parameters():
        param.requires_grad = True

    res = check_modules_to_save_trainable(model, ["head"])
    assert res["base_frozen"] is False
    assert res["valid"] is False


def test_compute_adapter_size_ratio():
    model = MockBaseModel()
    adapter_dict = {
        "head.weight": torch.randn(10, 32),
        "head.bias": torch.randn(10),
    }
    ratio = compute_adapter_size_ratio(model, adapter_dict)
    assert 0.0 < ratio < 1.0
