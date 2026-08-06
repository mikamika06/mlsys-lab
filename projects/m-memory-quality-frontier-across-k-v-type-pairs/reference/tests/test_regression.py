import sys

sys.path.insert(0, ".")
from kvquant.capacity import max_context_length
from kvquant.fallback import detect_fa_fallback
from kvquant.frontier import compute_pareto_frontier


def test_pareto_frontier_filters_dominated():
    cfg = {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}
    candidates = [
        {"k_type": "f16", "v_type": "f16", "perplexity_delta": 0.00},
        {"k_type": "q8_0", "v_type": "q8_0", "perplexity_delta": 0.02},
        {"k_type": "f32", "v_type": "f32", "perplexity_delta": 0.05},
    ]
    res = compute_pareto_frontier(cfg, 4096, candidates)
    types = [(r["k_type"], r["v_type"]) for r in res]
    assert ("f32", "f32") not in types


def test_fallback_detection():
    assert detect_fa_fallback("q8_0", "q4_0", 128)["fallback"] is True
    assert detect_fa_fallback("q8_0", "q8_0", 128)["fallback"] is False
    assert detect_fa_fallback("q8_0", "q8_0", 100)["fallback"] is True


def test_max_context_12gb_q8_0_accounts_for_fallback():
    budget = 12 * 1024 * 1024 * 1024
    base = 6 * 1024 * 1024 * 1024
    cfg = {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}

    c_good = max_context_length(budget, base, cfg, "q8_0", "q8_0")
    c_mismatch = max_context_length(budget, base, cfg, "q8_0", "q4_0")

    tb = {"q8_0": 1.0625, "q4_0": 0.5625}
    unit_mismatch_no_fb = cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * (tb["q8_0"] + tb["q4_0"])
    unpadded_max_c = int((budget - base) // unit_mismatch_no_fb)

    assert c_mismatch < unpadded_max_c
    assert c_good > 0
