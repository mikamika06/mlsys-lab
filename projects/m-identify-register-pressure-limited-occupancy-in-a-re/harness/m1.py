import ref


def check(workdir):
    from triton_prof.ncu import analyze_occupancy
    out = {"occupancy_matched": 0.0}
    ok = 0
    for case in ref.NCU_CASES:
        want = ref.analyze_occupancy(case)
        try:
            got = analyze_occupancy(case)
        except Exception:
            got = None
        if got == want:
            ok += 1
    out["occupancy_matched"] = float(ok)
    return out
