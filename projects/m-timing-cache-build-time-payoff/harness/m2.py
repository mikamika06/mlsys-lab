import ref

def check(workdir):
    from timingcache.optimization import find_knee
    out = {"knee_matched": 0.0}
    want = ref.find_knee(ref.LEVELS, ref.LATS, ref.COMPS)
    try:
        got = find_knee(ref.LEVELS, ref.LATS, ref.COMPS)
        if got == want:
            out["knee_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, expected {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {e}"
    return out
