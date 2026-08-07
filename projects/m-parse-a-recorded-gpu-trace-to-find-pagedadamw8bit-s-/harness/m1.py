import ref


def check(workdir):
    from gputrace import parse_trace_events

    out = {"traces_parsed": 0.0}
    ok = 0
    total = len(ref.TRACES)

    for i, trace in enumerate(ref.TRACES):
        want = ref.parse_trace_events(trace)
        try:
            got = parse_trace_events(trace)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"trace {i}: expected {want[:1]}, got {got[:1] if got else []}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"trace {i} raised {type(e).__name__}: {e}"

    if ok == total:
        out["traces_parsed"] = 1.0

    return out
