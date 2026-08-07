import ref

def check(workdir):
    from prefetch.histogram import compute_reuse_histogram
    trace = ref.generate_trace(123)
    want = ref.compute_reuse_histogram(trace, 50)
    got = compute_reuse_histogram(trace, 50)
    out = {"hist_match": 0.0}
    if got == want:
        out["hist_match"] = 1.0
    else:
        out["_note"] = f"histogram mismatch: got {got[:5]}, want {want[:5]}"
    return out
