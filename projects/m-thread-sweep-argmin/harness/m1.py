import ref

def check(workdir):
    from ortopt.sweep import thread_sweep
    out = {"optimal_thread_matched": 0.0}
    try:
        got = thread_sweep(ref.LATENCIES)
        want = ref.thread_sweep(ref.LATENCIES)
        if got == want:
            out["optimal_thread_matched"] = 1.0
        else:
            out["_note"] = f"got optimal threads {got}, want {want}"
    except Exception as e:
        out["_note"] = f"exception during thread_sweep: {type(e).__name__}: {str(e)[:120]}"
    return out
