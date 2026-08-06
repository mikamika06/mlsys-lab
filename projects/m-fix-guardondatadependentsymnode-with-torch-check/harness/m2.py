import ref


def check(workdir):
    from symfix.analysis import analyze_trace
    out = {"predictions_matched": 0.0}
    ok = 0
    for t in ref.TRACES:
        want = ref.predict_dynamic(t)
        got = analyze_trace(t)
        if got == want:
            ok += 1
    out["predictions_matched"] = 1.0 if ok == len(ref.TRACES) else 0.0
    return out
