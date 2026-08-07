import os
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))

    import ref
    from cache.analyzer import calculate_memory_footprint

    num_entries = 5000
    avg_key_len = 128
    avg_val_len = 512
    bytes_per_tok = 2

    expected = ref.ref_calculate_memory_footprint(num_entries, avg_key_len, avg_val_len, bytes_per_tok)

    out = {"memory_bytes_match": 0.0, "effective_cost_match": 0.0}
    try:
        actual = calculate_memory_footprint(num_entries, avg_key_len, avg_val_len, bytes_per_tok)
    except Exception:
        return out

    if actual.get("total_bytes") == expected["total_bytes"]:
        out["memory_bytes_match"] = 1.0

    if abs(actual.get("total_mb", -1) - expected["total_mb"]) < 1e-5:
        out["effective_cost_match"] = 1.0

    return out
