import os
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))

    import ref
    from cache.economics import evaluate_cache_viability

    trace = ref.generate_synthetic_trace(num_requests=1200, vocab_size=150, zipf_alpha=1.2, seed=789)
    capacity = 30
    compute_cost = 0.10
    memory_cost_per_entry = 0.50

    expected = ref.ref_evaluate_cache_viability(capacity, trace, compute_cost, memory_cost_per_entry)

    out = {"optimal_capacity_match": 0.0, "roi_match": 0.0}
    try:
        actual = evaluate_cache_viability(capacity, trace, compute_cost, memory_cost_per_entry)
    except Exception:
        return out

    if actual.get("should_enable") == expected["should_enable"] and abs(actual.get("net_savings", -999) - expected["net_savings"]) < 1e-5:
        out["optimal_capacity_match"] = 1.0

    if abs(actual.get("roi", -999) - expected["roi"]) < 1e-5 and abs(actual.get("breakeven_hit_rate", -1) - expected["breakeven_hit_rate"]) < 1e-5:
        out["roi_match"] = 1.0

    return out
