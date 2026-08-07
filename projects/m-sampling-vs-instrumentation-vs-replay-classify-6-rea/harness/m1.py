import ref


def check(workdir):
    from profiler.taxonomy import classify_mechanisms

    out = {"classification_matched": 0.0, "total": float(len(ref.TOOLS))}
    try:
        got = classify_mechanisms(ref.TOOLS)
        want = ref.classify_mechanisms(ref.TOOLS)
        if got == want:
            out["classification_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {e}"
    return out
