import sys
sys.path.insert(0, ".")
from leakdiag.cache import simulate_kv_cache
from leakdiag.loss import measure_loss_memory
from leakdiag.eval import check_activation_retention
import torch


def test_cache_resets_properly():
    mem_reset = simulate_kv_cache(100, reset=True)
    mem_leak = simulate_kv_cache(100, reset=False)
    assert mem_reset < mem_leak


def test_loss_memory_ratio():
    res = measure_loss_memory(50)
    assert res["ratio"] > 1.0


def test_eval_no_grad():
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    x = torch.randn(2, 10)
    res = check_activation_retention(model, x)
    assert res["cleared_with_nograd"] == 1.0
