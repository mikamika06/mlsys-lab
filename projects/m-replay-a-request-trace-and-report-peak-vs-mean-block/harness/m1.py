import ref

def check(workdir):
    from trace.replay import replay_trace
    traces = ref.get_traces()
    ok = 0
    out = {"timeline_matched": 0.0, "configs": float(len(traces))}
    for i, trace in enumerate(traces):
        want = ref.replay_trace(trace)
        try:
            got = replay_trace(trace)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"trace {i} raised {type(e).__name__}"
            continue
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got}, want {want}"
    out["timeline_matched"] = float(ok)
    return out
