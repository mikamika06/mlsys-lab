import os
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))

    import ref
    from cache.analyzer import analyze_trace

    trace = ref.generate_synthetic_trace(num_requests=500, vocab_size=50, zipf_alpha=1.1, seed=123)
    expected = ref.ref_analyze_trace(trace)

    out = {"hit_rate_match": 0.0, "uniques_match": 0.0}
    try:
        actual = analyze_trace(trace)
    except Exception:
        return out

    if abs(actual.get("hit_rate", -1) - expected["hit_rate"]) < 1e-6:
        out["hit_rate_match"] = 1.0

    if actual.get("unique_keys") == expected["unique_keys"]:
        out["uniques_match"] = 1.0

    return out
