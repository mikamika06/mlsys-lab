from slo.parse import parse_trace
from slo.classify import classify_violation

def classify_all(traces):
    """Classify all traces."""
    results = []
    for t in traces:
        p = parse_trace(t)
        cause = classify_violation(p, t)
        results.append({"req_id": t["req_id"], "cause": cause, **p})
    return results

def report_summary(classified):
    """Summarize violation causes."""
    counts = {"queueing": 0, "long-prefill": 0, "long-output": 0, "none": 0}
    for c in classified:
        counts[c["cause"]] += 1
    return counts
