import os
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))

    import ref
    from cache.economics import compute_net_savings

    hit_rate = 0.35
    total_requests = 10000
    compute_cost = 0.08
    memory_cost = 150.0

    expected = ref.ref_compute_net_savings(hit_rate, total_requests, compute_cost, memory_cost)

    out = {"savings_match": 0.0, "net_benefit_match": 0.0}
    try:
        actual = compute_net_savings(hit_rate, total_requests, compute_cost, memory_cost)
    except Exception:
        return out

    if abs(actual.get("compute_saved", -1) - expected["compute_saved"]) < 1e-5:
        out["savings_match"] = 1.0

    if abs(actual.get("net_savings", -999) - expected["net_savings"]) < 1e-5 and actual.get("is_profitable") == expected["is_profitable"]:
        out["net_benefit_match"] = 1.0

    return out
