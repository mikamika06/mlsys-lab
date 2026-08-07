import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from optmem.state import estimate_optimizer_state_bytes, calculate_model_optimizer_footprint
from optmem.zerograd import profile_zerograd_allocation


def test_optimizer_state_memory_calculation():
    params = [torch.randn(10, 20, dtype=torch.float32), torch.randn(5, dtype=torch.float32)]
    total_numel = 200 + 5
    expected_sgd = 0
    expected_momentum = total_numel * 4
    expected_adam = 2 * total_numel * 4

    assert estimate_optimizer_state_bytes(params, "sgd", initialized=True) == expected_sgd
    assert estimate_optimizer_state_bytes(params, "momentum", initialized=True) == expected_momentum
    assert estimate_optimizer_state_bytes(params, "adam", initialized=True) == expected_adam
    assert estimate_optimizer_state_bytes(params, "adam", initialized=False) == 0

    footprint = calculate_model_optimizer_footprint(params, ["sgd", "adam"])
    assert footprint["sgd"]["state_bytes"] == 0
    assert footprint["adam"]["state_bytes"] == expected_adam


def test_zerograd_set_to_none_reduces_allocations():
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x = torch.randn(4, 16)
    res = profile_zerograd_allocation(model, x)

    assert res["zero_fill_count"] > 0
    assert res["none_fill_count"] == 0
    assert res["zero_fill_bytes"] > 0
    assert res["none_fill_bytes"] == 0
    assert res["allocated_bytes_saved"] == res["zero_fill_bytes"]
