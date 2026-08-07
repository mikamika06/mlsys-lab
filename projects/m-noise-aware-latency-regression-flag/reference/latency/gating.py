from latency.detector import classify_run

def validate_execution_mode(latencies, baseline_med, baseline_mad, max_rel_diff):
    status = classify_run(latencies, baseline_med, baseline_mad, max_rel_diff)
    if status == "silent_eager_fallback":
        return False
    return True
