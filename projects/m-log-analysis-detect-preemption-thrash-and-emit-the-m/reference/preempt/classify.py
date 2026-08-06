def classify_traces(traces):
    """Classifies execution traces into capacity_bound or arrival_bound."""
    results = []
    for trace in traces:
        queue = trace.get("queue_depth", [])
        kv = trace.get("kv_usage_frac", [])
        is_cap = False
        for q, k in zip(queue, kv):
            if k >= 0.85 and q > 0:
                is_cap = True
                break
        results.append("capacity_bound" if is_cap else "arrival_bound")
    return results
