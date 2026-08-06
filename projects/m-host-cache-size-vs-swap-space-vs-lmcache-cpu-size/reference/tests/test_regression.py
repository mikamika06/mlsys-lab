import sys

sys.path.insert(0, ".")
from memplan.config import compute_allocations, evaluate_feasibility, optimization_schedule
from memplan.evaluator import calculate_overhead, verify_memory_bounds

CONFIG = {
    "host_cache_size": 1024,
    "swap_space": 2048,
    "lmcache_cpu_size": 512,
    "block_size": 16
}


def test_allocations_non_negative():
    res = compute_allocations(CONFIG)
    assert res["host_cache_bytes"] >= 0
    assert res["swap_bytes"] >= 0
    assert res["lmcache_bytes"] >= 0


def test_feasibility_logic():
    assert evaluate_feasibility(CONFIG, 100000000) is True
    assert evaluate_feasibility(CONFIG, 100) is False


def test_overhead_scaling():
    ov = calculate_overhead(1000, 1000, 1000)
    assert ov > 0


def test_schedule_monotonicity():
    s = optimization_schedule(512, 5)
    assert all(s[i] <= s[i + 1] for i in range(len(s) - 1))
    assert s[0] == 512
