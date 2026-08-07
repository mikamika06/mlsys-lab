import ref

def check(workdir):
    from servetrace.recorder import record_trace
    events = ref.generate_trace_data()
    total_time = max(e[0] for e in events)
    want = ref.record_trace(events, total_time)
    try:
        got = record_trace(events, total_time)
    except Exception as e:
        return {"trace_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    if got == want:
        return {"trace_matched": 1.0}
    return {"trace_matched": 0.0, "_note": f"got {got[:3]}, want {want[:3]}"}
