import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from specdec.metrics import compute_speedup
    except Exception as e:
        return {"speedup_match": 0.0, "trace_metrics_exact": 0.0, "_note": f"Import failed: {e}"}

    speedup_ok = True
    exact_ok = True

    for case in ref.M2_TRACES:
        want = ref.compute_speedup(case["trace"], case["k"], case["draft_cost"], case["target_cost"])
        got = compute_speedup(case["trace"], case["k"], case["draft_cost"], case["target_cost"])

        if not isinstance(got, dict):
            return {"speedup_match": 0.0, "trace_metrics_exact": 0.0, "_note": "compute_speedup must return a dict"}

        for k, v in want.items():
            if k not in got:
                exact_ok = False
                break
            if abs(float(got[k]) - float(v)) > 1e-5:
                speedup_ok = False
                exact_ok = False

    return {
        "speedup_match": 1.0 if speedup_ok else 0.0,
        "trace_metrics_exact": 1.0 if exact_ok else 0.0,
    }
