import sys
sys.path.insert(0, ".")
from servemetrics.bench import compute_tpot_overhead
from servemetrics.engine import simulate_generation

CONFIG = {"schema_complexity": "medium", "vocab_size": 32000, "output_tokens": 64}

def test_latency_ratio_greater_than_one():
    res = compute_tpot_overhead(CONFIG, seed=42)
    assert res["latency_ratio"] > 1.0, f"Expected guided decoding to increase TPOT ratio above 1.0, got {res['latency_ratio']}"

def test_simulation_length():
    latencies = simulate_generation(CONFIG, guided=False, seed=42)
    assert len(latencies) == CONFIG["output_tokens"]

def test_tpot_values_positive():
    res = compute_tpot_overhead(CONFIG, seed=42)
    assert res["mean_unconstrained_tpot"] > 0
    assert res["mean_guided_tpot"] > 0
