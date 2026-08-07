import ref

def check(workdir):
    from prefetch.detector import detect_wasted
    trace = ref.generate_trace(123)
    want = ref.detect_wasted(trace, 15)
    got = detect_wasted(trace, 15)
    out = {"wasted_match": 0.0}
    if got == want:
        out["wasted_match"] = 1.0
    else:
        out["_note"] = f"wasted prefetch mismatch: got {got}, want {want}"
    return out
