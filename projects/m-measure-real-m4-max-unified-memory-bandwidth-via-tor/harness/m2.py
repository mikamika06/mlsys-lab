import ref

def check(workdir):
    from mprofiler.memory import build_memory_timeline
    out = {"timeline_matched": 0.0}
    allocs, drivers = ref.TIMELINE_INPUTS
    got = build_memory_timeline(allocs, drivers)
    want = [{"allocated": a, "driver": d, "diff": d - a} for a, d in zip(allocs, drivers)]
    if got == want:
        out["timeline_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
