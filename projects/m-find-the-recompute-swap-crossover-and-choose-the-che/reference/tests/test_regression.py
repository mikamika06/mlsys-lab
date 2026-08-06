import sys
sys.path.insert(0, ".")
from preemption.policy import choose_cheaper_mode
from preemption.crossover import find_crossover

def test_policy_valid_return():
    model = {"layers": 32, "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128}
    system = {"swap_bandwidth_gbps": 32.0, "flops_per_token": 1.5e9, "block_size": 16}
    mode = choose_cheaper_mode(512, model, system)
    assert mode in ("swap", "recompute")

def test_crossover_is_positive_integer():
    model = {"layers": 32, "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128}
    system = {"swap_bandwidth_gbps": 32.0, "flops_per_token": 1.5e9, "block_size": 16}
    c = find_crossover(model, system)
    assert isinstance(c, int)
    assert c > 0

def test_short_context_favors_swap():
    model = {"layers": 32, "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128}
    system = {"swap_bandwidth_gbps": 64.0, "flops_per_token": 2.0e9, "block_size": 16}
    mode = choose_cheaper_mode(16, model, system)
    assert mode == "swap"
