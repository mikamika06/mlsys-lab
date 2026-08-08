def classify_violation(parsed, trace):
    """Classify SLO violation cause."""
    if not parsed["is_violation"]:
        return "none"
    q = parsed["queue_time"]
    p = parsed["prefill_time"]
    o = parsed["output_time"]
    vals = {"queueing": q, "long-prefill": p, "long-output": o}
    return max(vals, key=vals.get)
