import sys

sys.path.insert(0, ".")
from moe_sim.metrics import latency, sweep_configs, active_fraction

CFG = {
    "layers": 32,
    "base_mem": 200,
    "expert_mem": 100,
    "n_experts": 8,
    "top_k": 2,
    "time_base_gpu": 2,
    "time_base_cpu": 20,
    "time_exp_gpu": 1,
    "time_exp_cpu": 15
}

def test_latency_probability_routing():
    lat = latency(CFG, 32, 4)
    assert abs(lat - 576.0) < 1e-5, f"Expected 576, got {lat}"

def test_sweep_never_exceeds_vram():
    max_v = 15000
    res = sweep_configs(CFG, max_v)
    for conf in res:
        assert conf["vram"] <= max_v, "VRAM limit exceeded"
        assert conf["ngl"] <= CFG["layers"], "Cannot exceed total layers"

def test_active_fraction_is_bounded():
    frac = active_fraction(CFG)
    assert 0 < frac < 1.0, "Fraction must be between 0 and 1"
