import ref

def check(workdir):
    from timingcache.cache import compute_payoff
    out = {"payoff_matched": 0.0}
    want = ref.compute_payoff(ref.BUILD_TIMES_NC, ref.BUILD_TIMES_WC)
    try:
        got = compute_payoff(ref.BUILD_TIMES_NC, ref.BUILD_TIMES_WC)
        if len(got) == len(want) and all(abs(a - b) < 1e-4 for a, b in zip(got, want)):
            out["payoff_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, expected {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {e}"
    return out
