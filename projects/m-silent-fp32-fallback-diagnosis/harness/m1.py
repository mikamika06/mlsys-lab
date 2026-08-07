import ref


def check(workdir):
    from fallbackdiag.diagnosis import detect_fallbacks
    out = {"fallbacks_matched": 0.0}
    ok = 0
    for i, spec in enumerate(ref.MODELS):
        want = ref.detect_fallbacks(spec)
        got = detect_fallbacks(spec)
        if got == want:
            ok += 1
    out["fallbacks_matched"] = float(ok == len(ref.MODELS))
    return out
