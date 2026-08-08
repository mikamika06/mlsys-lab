def parse_trace(trace):
    """Parse raw trace into latency components."""
    q = trace["start_time"] - trace["arrival_time"]
    p = trace["first_token_time"] - trace["start_time"]
    o = trace["completion_time"] - trace["first_token_time"]
    tot = trace["completion_time"] - trace["arrival_time"]
    return {
        "req_id": trace["req_id"],
        "queue_time": round(q, 4),
        "prefill_time": round(p, 4),
        "output_time": round(o, 4),
        "total_latency": round(tot, 4),
        "is_violation": tot > trace["slo_target"]
    }
