import ref


def check(workdir):
    from gputrace import parse_trace_events, find_spillover_spike

    out = {"spillover_index_matched": 0.0, "ratio_matched": 0.0}
    idx_ok = 0
    ratio_ok = 0
    total = len(ref.TRACES)

    for i, trace in enumerate(ref.TRACES):
        events = ref.parse_trace_events(trace)
        want = ref.find_spillover_spike(events)
        try:
            got = find_spillover_spike(events)
            if got.get("argmin_index") == want.get("argmin_index"):
                idx_ok += 1
            if abs(got.get("max_ratio", 0.0) - want.get("max_ratio", 0.0)) < 1e-5:
                ratio_ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"trace {i} raised {type(e).__name__}: {e}"

    if idx_ok == total:
        out["spillover_index_matched"] = 1.0
    if ratio_ok == total:
        out["ratio_matched"] = 1.0

    return out
