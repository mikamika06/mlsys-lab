import ref


def check(workdir):
    from runner.diagnose import analyze_queue_and_503

    trace = ref.generate_trace_data()
    res = analyze_queue_and_503(queue_capacity=2, request_trace=trace)

    out = {
        "diagnosed_503_cause": 0.0,
        "hol_blocking_detected": 0.0,
    }

    if res and "queue_overflow" in res.get("cause_503", ""):
        out["diagnosed_503_cause"] = 1.0

    events = res.get("hol_blocking_events", []) if res else []
    if len(events) >= 1 and events[0].get("blocking_request_id") == "r1":
        out["hol_blocking_detected"] = 1.0

    return out
