import ref

def check(workdir):
    from trace.replay import replay_trace
    from trace.metrics import compute_occupancy
    traces = ref.get_traces()
    ok = 1
    out = {"metrics_matched": 0.0}
    for i, trace in enumerate(traces):
        timeline = ref.replay_trace(trace)
        want = ref.compute_occupancy(timeline)
        try:
            got_timeline = replay_trace(trace)
            got = compute_occupancy(got_timeline)
        except Exception as e:
            ok = 0
            out["_note"] = f"trace {i} raised {type(e).__name__}"
            break
        if got["peak"] != want["peak"] or abs(got["mean"] - want["mean"]) > 1e-5:
            ok = 0
            out["_note"] = f"trace {i}: got {got}, want {want}"
            break
    out["metrics_matched"] = float(ok)
    return out
