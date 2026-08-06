import numpy as np


def analyze_metrics(trace_data):
    qps = trace_data["qps"]
    max_seqs = trace_data["max_seqs"]
    events = trace_data["events"]
    queue_waits = trace_data["queue_waits"]
    mean_wait = float(np.mean(queue_waits))
    p99_wait = float(np.percentile(queue_waits, 99))
    is_preemption = bool(events > 10)
    return {
        "mean_queue_wait": mean_wait,
        "p99_queue_wait": p99_wait,
        "preemption_events": events,
        "is_preemption_dominant": is_preemption,
        "recommended_max_seqs": int(max_seqs if not is_preemption else max_seqs + 64)
    }
