import sys
sys.path.insert(0, ".")
from moe_bench.concurrent import max_concurrent_experts

CONFIG = {
    "model_name": "Test-MoE",
    "total_params": 50000000000,
    "num_experts": 8,
    "active_experts": 2,
    "expert_size_bytes": 5000000000,
    "base_size_bytes": 10000000000,
}


def test_zero_ceiling():
    assert max_concurrent_experts(CONFIG, 5000000000) == 0


def test_exact_base():
    assert max_concurrent_experts(CONFIG, 10000000000) == 0


def test_ample_ceiling():
    assert max_concurrent_experts(CONFIG, 100000000000) == 8


def test_partial_ceiling():
    val = max_concurrent_experts(CONFIG, 25000000000)
    assert val == 3
