import ref

def check(workdir):
    from bench.core import robust_summary
    sample = [1.0, 1.1, 1.05, 10.0, 1.02]
    want = ref.robust_summary(sample)
    try:
        got = robust_summary(sample)
    except Exception as e:
        return {"robust_matched": 0.0, "_note": f"raised {e}"}

    out = {"robust_matched": 0.0}
    if isinstance(got, dict) and all(k in got for k in ("mean", "median", "p90")):
        if abs(got["median"] - want["median"]) < 1e-5:
            out["robust_matched"] = 1.0
        else:
            out["_note"] = f"got median {got['median']}, want {want['median']}"
    else:
        out["_note"] = f"missing keys in summary: {got}"
    return out
