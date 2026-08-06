import sys

sys.path.insert(0, ".")
from eagle.config import build_eagle_config
from eagle.server import run_server_simulation
from eagle.metrics import compute_eagle_metrics


def test_acceptance_rate_bounds():
    run_cfg = {
        "accepted": 50,
        "total": 100,
        "baseline_tpot": 30.0,
        "eagle_tpot": 15.0
    }
    sim = run_server_simulation(run_cfg)
    metrics = compute_eagle_metrics(run_cfg, sim)
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0


def test_latency_ratio_positive():
    run_cfg = {
        "accepted": 50,
        "total": 100,
        "baseline_tpot": 30.0,
        "eagle_tpot": 15.0
    }
    sim = run_server_simulation(run_cfg)
    metrics = compute_eagle_metrics(run_cfg, sim)
    assert metrics["latency_ratio"] > 0.0
    assert metrics["tpot_gain"] > 0.0


def test_config_keys():
    cfg = {
        "model": "llama-8b",
        "top_k": 4,
        "depth": 3,
        "num_speculative_tokens": 5
    }
    res = build_eagle_config(cfg)
    assert "speculative_model" in res
    assert "tree_config" in res
