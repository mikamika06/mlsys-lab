import sys
sys.path.insert(0, ".")
from trtopt.tune import tune_max_tokens
from trtopt.simulate import simulate_performance
from trtopt.config import load_default_config


def test_slo_compliance():
    cfg = load_default_config()
    best = tune_max_tokens(
        cfg["candidate_tokens"],
        cfg["slo_ttft"],
        "continuous",
        cfg["arrival_rate"],
        cfg["prefill_lengths"]
    )
    t, _ = simulate_performance(best, "continuous", cfg["arrival_rate"], cfg["prefill_lengths"])
    assert t <= cfg["slo_ttft"]
    assert best in cfg["candidate_tokens"]


def test_argmin_selection():
    tokens = [256, 512, 1024]
    best = tune_max_tokens(tokens, 1000.0, "static", 10.0, [100])
    assert best > 0
