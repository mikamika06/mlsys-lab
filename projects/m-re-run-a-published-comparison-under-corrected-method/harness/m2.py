import ref
import numpy as np


def check(workdir):
    from runner_audit.audit import compute_required_repeats
    data = ref.make_test_cases()
    latencies = data["latencies"]

    out = {"rel_err": 1.0}
    try:
        got_n = compute_required_repeats(latencies, target_rel_error=0.05, confidence=0.95)
        arr = np.array(latencies, dtype=float)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        z = 1.96
        computed_err = (z * std) / np.sqrt(got_n) / mean
        out["rel_err"] = float(computed_err)
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {e}"
    return out
