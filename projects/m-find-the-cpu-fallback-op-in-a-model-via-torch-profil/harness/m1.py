import ref


def check(workdir):
    from mps_diag.profiler import find_fallback_ops

    events = ref.generate_mock_trace_events()
    want = ref.identify_fallbacks(events)
    try:
        got = find_fallback_ops(events)
        got_sorted = sorted(list(set(got or [])))
    except Exception as e:
        return {"fallbacks_identified": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    if got_sorted == want:
        return {"fallbacks_identified": 1.0}
    return {"fallbacks_identified": 0.0, "_note": f"got {got_sorted}, want {want}"}
