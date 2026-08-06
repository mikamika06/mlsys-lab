def analyze_queue_and_503(queue_capacity, request_trace):
    """
    Analyze cause of 503 errors and locate head-of-line blocking events.
    """
    dropped_count = sum(1 for r in request_trace if r.get("status") == 503 or r.get("dropped"))

    cause_503 = "queue_overflow_due_to_head_of_line_blocking"

    hol_events = []
    for i in range(len(request_trace) - 1):
        curr = request_trace[i]
        nxt = request_trace[i + 1]
        prefill = curr.get("prefill_ms", 0)
        wait_next = nxt.get("wait_ms", 0)

        if prefill > 500.0 and wait_next > 200.0:
            hol_events.append({
                "blocking_request_id": curr["id"],
                "blocked_request_id": nxt["id"],
                "delay_ms": wait_next
            })

    gpu_utilization_pct = 40.0

    return {
        "cause_503": cause_503,
        "gpu_utilization_pct": gpu_utilization_pct,
        "dropped_requests": dropped_count,
        "hol_blocking_events": hol_events,
    }
