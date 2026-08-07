import sys
sys.path.insert(0, ".")

from ktrans.placement import reconstruct_placement
from ktrans.cache import simulate_lru_cache
from ktrans.offload import evaluate_offload_latency


def test_placement_budget_constraint():
    num_layers = 2
    num_experts = 4
    expert_bytes = 100
    vram_budget = 250
    freq_log = {(0, 0): 10, (0, 1): 8, (1, 0): 15, (1, 1): 5}

    res = reconstruct_placement(num_layers, num_experts, expert_bytes, vram_budget, freq_log)
    total_gpu_experts = sum(len(res[l]["gpu"]) for l in res)
    assert total_gpu_experts * expert_bytes <= vram_budget
    assert len(res[0]["gpu"]) + len(res[0]["cpu"]) == num_experts
    assert len(res[1]["gpu"]) + len(res[1]["cpu"]) == num_experts


def test_lru_cache_bounds():
    trace = [1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2]
    rate = simulate_lru_cache(2, trace)
    assert 0.0 <= rate <= 1.0
    assert simulate_lru_cache(0, trace) == 0.0


def test_offload_latency_positive():
    res = evaluate_offload_latency(32, 128, 0.001, 0.005, 0.002, 16)
    assert res["offload_all_latency"] > 0
    assert res["offload_split_latency"] > 0
    assert res["speedup"] > 0
