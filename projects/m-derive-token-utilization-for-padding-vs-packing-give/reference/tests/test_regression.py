import torch
import torch.nn as nn
from packutil.metrics import compute_utilization
from packutil.train import run_dummy_finetune
from packutil.verify import verify_adapter_only

def test_utilization_bounds():
    res = compute_utilization([10, 20, 30], 100)
    assert 0.0 < res["padding_utilization"] <= 1.0
    assert 0.0 < res["packing_utilization"] <= 1.0
    assert res["packing_utilization"] >= res["padding_utilization"]

def test_finetune_loss():
    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [torch.randn(5, 10) for _ in range(5)]
    res = run_dummy_finetune(model, optimizer, data)
    assert res["loss_decreased"] is True

def test_adapter_verification():
    base = {"weight": torch.zeros(2, 2), "lora_adapter": torch.ones(2, 2)}
    post = {"weight": torch.zeros(2, 2), "lora_adapter": torch.full((2, 2), 0.5)}
    assert verify_adapter_only(base, post) is True
    post_bad = {"weight": torch.ones(2, 2), "lora_adapter": torch.full((2, 2), 0.5)}
    assert verify_adapter_only(base, post_bad) is False
