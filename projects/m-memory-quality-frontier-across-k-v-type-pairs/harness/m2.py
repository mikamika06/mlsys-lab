import ref


def check(workdir):
    from kvquant.fallback import detect_fa_fallback

    out = {"fallbacks_matched": 0.0}
    ok = 0
    total = len(ref.SAMPLE_FALLBACK_CASES)
    for case in ref.SAMPLE_FALLBACK_CASES:
        want = ref.detect_fa_fallback(case["k_type"], case["v_type"], case["head_dim"])
        got = detect_fa_fallback(case["k_type"], case["v_type"], case["head_dim"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {case}: got {got}, want {want}"
    if ok == total:
        out["fallbacks_matched"] = 1.0
    return out
