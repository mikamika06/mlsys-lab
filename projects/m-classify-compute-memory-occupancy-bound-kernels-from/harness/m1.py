import ref

def check(workdir):
    from trace_parser.ncu import classify_ncu
    from trace_parser.proton import analyze_proton

    out = {"ncu_match": 0.0, "proton_match": 0.0}

    ncu_got = classify_ncu(ref.NCU_FIXTURE)
    ncu_want = ref.classify_ncu(ref.NCU_FIXTURE)
    if ncu_got == ncu_want:
        out["ncu_match"] = 1.0

    p_got = analyze_proton(ref.PROTON_FIXTURE)
    p_want = ref.analyze_proton(ref.PROTON_FIXTURE)
    if p_got and p_want and set(p_got.keys()) == set(p_want.keys()):
        diffs = [abs(p_got[k] - p_want[k]) for k in p_want]
        if max(diffs) < 1e-5:
            out["proton_match"] = 1.0

    return out
