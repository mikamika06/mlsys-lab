import ref

def check(workdir):
    from servetrace.compare import compare_traces
    events = ref.generate_trace_data()
    total_time = max(e[0] for e in events)
    t_trace = ref.record_trace(events, total_time)
    r_trace = ref.record_trace(events, total_time)
    want = ref.compare_traces(t_trace, r_trace)
    try:
        got = compare_traces(t_trace, r_trace)
    except Exception as e:
        return {"comparison_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    if got == want:
        return {"comparison_matched": 1.0}
    return {"comparison_matched": 0.0, "_note": f"got {got}, want {want}"}
