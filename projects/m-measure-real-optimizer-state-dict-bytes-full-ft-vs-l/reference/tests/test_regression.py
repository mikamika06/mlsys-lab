import sys
import torch

sys.path.insert(0, ".")
from optmem.qlora import verify_qlora_optimizer_clean


def test_catches_frozen_state_leak():
    model = torch.nn.Linear(10, 10)
    model.weight.requires_grad = False

    opt = torch.optim.Adam(model.parameters())
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()

    assert verify_qlora_optimizer_clean(model, opt) is False
