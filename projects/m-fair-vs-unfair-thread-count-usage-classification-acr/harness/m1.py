import ref


def check(workdir):
    from threadperf.classifier import classify_runs

    out = {"runs_classified": 0.0}
    want = ref.classify_runs(ref.RUNS)
    try:
        got = classify_runs(ref.RUNS)
    except Exception as e:
        out["_note"] = f"classifier raised exception: {type(e).__name__}: {str(e)[:100]}"
        return out

    if not isinstance(got, list):
        out["_note"] = f"expected list, got {type(got)}"
        return out

    ok = 0
    for i, (w, g) in enumerate(zip(want, got)):
        if w == g:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"run {i}: got {g}, want {w}"

    out["runs_classified"] = float(ok)
    return out
