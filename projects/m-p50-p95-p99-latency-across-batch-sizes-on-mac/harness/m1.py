import ref
import numpy as np


def check(workdir):
    try:
        from latbench.stats import compute_percentiles, analyze_batch_latencies
    except ImportError as e:
        return {"stats_matched": 0.0, "_note": f"Import error: {e}"}

    data = ref.generate_synthetic_data(seed=123)
    p_data = data["profile_data"]

    want_summary = ref.analyze_batch_latencies(p_data)
    try:
        got_summary = analyze_batch_latencies(p_data)
    except Exception as e:
        return {"stats_matched": 0.0, "_note": f"Execution error: {e}"}

    if not isinstance(got_summary, dict):
        return {"stats_matched": 0.0, "_note": "Output is not a dict"}

    matched = True
    for b, want_stats in want_summary.items():
        if b not in got_summary:
            matched = False
            break
        got_stats = got_summary[b]
        for metric in ["p50", "p95", "p99"]:
            if metric not in got_stats:
                matched = False
                break
            if not np.isclose(got_stats[metric], want_stats[metric], rtol=1e-3):
                matched = False
                break
        if not matched:
            break

    return {
        "stats_matched": 1.0 if matched else 0.0,
        "_note": "Stats matched reference" if matched else "Mismatch in percentile statistics"
    }
